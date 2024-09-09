import pandas as pd
import numpy as np
from io import StringIO
from flask import Flask, jsonify, request
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
import os
import pickle

JSON_ORIENT = "split"

@dataclass
class metaDataStorage:
    building_id : str
    smartmeter_ids : list = field(default_factory=list)
    min_values : dict = field(default_factory=dict)
    max_values : dict = field(default_factory=dict)

class demandForecastDataManager:
    """
    Manages energy consumption data from Genia instances. 

    Tasks:
        - Accept and store new measurement data
    """
    def __init__(self):
        # setup db
        self.db_path = "db/"
        self.data_path = self.db_path + "data/"
        self.smartmeter_path = self.db_path + "smartmeter_ids.pkl"
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path, exist_ok=True)
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path, exist_ok=True)

        # setup app
        self.app = Flask(import_name=__name__)
        self.configure_routes()
        self.last_restart = datetime.now()

        # reload identifiers
        self.meta_data = {} # dictionary 
        self.reload_identifiers()

    def configure_routes(self):
        """Sets up flask routes."""
        @self.app.route("/", methods=['GET'])
        def index():
            return jsonify({'message': 'Index of demandForecastDataManager.'}), 200
        
        @self.app.route("/getHealth", methods=['GET'])
        def get_health():
            return jsonify({'status':'up', 'lastRestart': self.last_restart.isoformat()}), 200
        
        @self.app.route("/getIds", methods=['GET'])
        def get_ids():
            """Return all building identifiers in database."""
            return jsonify({'buildingIds': list(self.meta_data.keys())})
        
        @self.app.route("/getSmartmeterIds", methods=['GET'])
        def get_smartmeter_ids():
            """Returns smartmeter ids for building id."""
            # parse and check   
            building_id = request.headers.get("buildingIdentifier")
            valid, message = self.check_building_identifier_db(building_identifier=building_id)
            
            if not valid:
                return jsonify({'message': message}), 400
            
            # get smartmeter ids
            smartmeter_ids = self.meta_data[building_id].smartmeter_ids

            return jsonify({'smartmeterIdentifiers': smartmeter_ids})

        ## add new id
        @self.app.route("/addId", methods=['POST'])
        def add_id():
            # parse and check identifiers from request
            building_identifier = request.headers.get("buildingIdentifier")
            smartmeter_identifiers = request.headers.get("smartmeterIdentifiers")
            if building_identifier is None or smartmeter_identifiers is None:
                return jsonify({'message': 'Building Identifier or Smartmeter identifier is None'}), 400
            smartmeter_identifiers = smartmeter_identifiers.split(",")

            # check building identifier
            valid, message = self.check_building_identifier_db(building_identifier=building_identifier)
            if valid:
                return jsonify({'message': "Building identifier already exists in database. Use the /updateId endpoint."}), 400
            
            # check smartmeter identifiers
            valid, message = self.check_smartmeter_identifiers_dtype(smartmeter_identifiers=smartmeter_identifiers)
            if not valid:
                return jsonify({'message': message}), 400

            # store new identifiers
            meta_data_storage = metaDataStorage(
                building_id=building_identifier,
                smartmeter_ids=smartmeter_identifiers,
                min_values={smartmeter_identifier: float('inf') for smartmeter_identifier in smartmeter_identifiers},
                max_values={smartmeter_identifier: float('-inf') for smartmeter_identifier in smartmeter_identifiers}
            )
            self.meta_data[building_identifier] = meta_data_storage

            # create and store dataframe
            columns = ['timestamp']
            for smartmeter_identifier in smartmeter_identifiers:
                columns.append(f"{smartmeter_identifier}_delta_energy")
            data = pd.DataFrame(columns=columns)
            data.to_feather(f"{self.data_path}{building_identifier}.feather")

            # store identifiers
            self.store_identifiers()

            return jsonify({'message': f'Created Database entry for {building_identifier}'}), 200
        
        # update ids
        @self.app.route("/updateId", methods=['PUT'])
        def update_id():
            # parse and check identifiers
            building_identifier = request.headers.get("buildingIdentifier")
            smartmeter_identifiers = request.headers.get("smartmeterIdentifiers")
            smartmeter_identifiers = smartmeter_identifiers.split(",")

            # check if building identifier in database
            valid, message = self.check_building_identifier_db(building_identifier=building_identifier)
            if not valid:
                return jsonify({'message': message}), 400
            
            # check if smartmeter identifiers in database
            valid, message = self.check_smartmeter_identifiers_db(
                building_identifier=building_identifier,
                smartmeter_identifiers=smartmeter_identifiers)
            if valid:
                return jsonify({'message': 'Smartmeter identifiers are already in database.'}), 400
            ids_not_in_db = message['ids_not_in_db']
            
            # add new smartmeter identifiers to database
            self.meta_data[building_identifier].smartmeter_ids.extend(ids_not_in_db)
            self.meta_data[building_identifier].min_values.update({sm_id: 0.0 for sm_id in ids_not_in_db})
            self.meta_data[building_identifier].max_values.update({sm_id: 0.0 for sm_id in ids_not_in_db})

            # add new columns to database
            try:
                data = pd.read_feather(f"{self.data_path}/{building_identifier}.feather")
                for smartmeter_identifier in ids_not_in_db:
                    data[smartmeter_identifier + "_delta_energy"] = np.nan
            except FileNotFoundError:
                columns = ['timestamp']
                columns.extend(smartmeter_identifiers)
                data = pd.DataFrame(columns=columns)

            data.to_feather(f"{self.data_path}{building_identifier}.feather")

            # store identifiers
            self.store_identifiers()

            return jsonify({'message': f"Updated Smartmeter Ids for {building_identifier}"}), 200
        
        @self.app.route("/deleteSmartmeterId", methods=['DELETE'])
        def delete_smartmeter_id():
            # parse and check identifiers
            building_identifier = request.headers.get("buildingIdentifier")
            smartmeter_identifiers = request.headers.get("smartmeterIdentifiers")
            smartmeter_identifiers = smartmeter_identifiers.split(",")

            # check if smartmeter identifier in database
            valid, message = self.check_smartmeter_identifiers_db(building_identifier=building_identifier, smartmeter_identifiers=smartmeter_identifiers)
            if not valid:
                try:
                    return jsonify({'message': f'The following ids are not in the db {message["ids_not_in_db"]}'}), 400
                except TypeError:
                    return jsonify({'message': message}), 400
            
            # remove items
            for smartmeter_identifier in smartmeter_identifiers:
                self.meta_data[building_identifier].smartmeter_ids.remove(smartmeter_identifier)
                del self.meta_data[building_identifier].min_values[smartmeter_identifier]
                del self.meta_data[building_identifier].max_values[smartmeter_identifier]

            # delete columns
            data = pd.read_feather(f"{self.data_path}{building_identifier}.feather")
            smartmeter_identifiers = [smartmeter_identifier + "_delta_energy" for smartmeter_identifier in smartmeter_identifiers]
            data.drop(columns=smartmeter_identifiers, inplace=True)
            data.to_feather(f"{self.data_path}{building_identifier}.feather")

            return jsonify({'message': 'Smartmeter Identifiers successfully removed from database.'}), 200

        @self.app.route("/deleteBuildingId", methods=['DELETE'])
        def delete_building_id():
            # parse and check identifier
            building_identifier = request.headers.get("buildingIdentifier")

            # check if building identifier in database
            valid, message = self.check_building_identifier_db(building_identifier=building_identifier)
            if not valid:
                return jsonify({'message': message}), 400
            
            # delete building identifier
            del self.meta_data[building_identifier]

            # delete dataframe
            os.remove(f"{self.data_path}{building_identifier}.feather" )

            return jsonify({'message':'Building identifier successfully removed from database.'}), 200
        
        @self.app.route("/getMeasurements", methods=['GET'])
        def get_measurements():
            building_identifier = request.headers.get("buildingIdentifier")
            start = request.headers.get("start")
            end = request.headers.get("end")

            # check start and end
            if start is None or end is None:
                return jsonify({'message': f"Start or end date is not set."}), 400
            
            # convert start and end
            try:
                start, end = datetime.fromisoformat(start), datetime.fromisoformat(end)
            except ValueError:
                return jsonify({'message': 'Start or end is invalid iso string'}), 400

            # check building identifier
            if building_identifier not in self.meta_data.keys():
                return jsonify({'message': f'Building identifier {building_identifier} does not exist in database.'}), 404
            
            data = pd.read_feather(f"{self.data_path}{building_identifier}.feather")
            data = data[(data.timestamp >= start) & (data.timestamp <= end)]
            data_json = data.to_json(orient=JSON_ORIENT)

            return jsonify({'data': data_json}), 200
        
        @self.app.route("/getMinMax", methods=['GET'])
        def get_min_max():
            building_id = request.headers.get("buildingIdentifier")
            valid, message = self.check_building_identifier_db(building_identifier=building_id)

            if not valid:
                return jsonify({'message': message}), 400
            
            min_values = self.meta_data[building_id].min_values
            max_values = self.meta_data[building_id].max_values

            return jsonify({
                'min': min_values,
                'max': max_values
            }), 200

        @self.app.route("/postMeasurements", methods=['POST'])
        def post_measurements():
            building_identifier = request.headers.get('buildingIdentifier')
            smartmeter_identifiers = request.headers.get('smartmeterIdentifiers')
            if building_identifier is None or smartmeter_identifiers is None:
                return jsonify({'message': 'Either buildingIdentifier or smartmeterIdentifier is None'}), 400
            smartmeter_identifiers = smartmeter_identifiers.split(",")
            data_json = request.json

            # check building identifiers and smartmeter identifiers
            valid, message = self.check_smartmeter_identifiers_db(building_identifier=building_identifier, smartmeter_identifiers=smartmeter_identifiers)
            if not valid:
                return jsonify({'message': message}), 400
            
            if data_json is None:
                return jsonify({'message': "Missing data."}), 400
            
            # convert to dataframe
            dtype = {'timestamp':'str'}
            for smartmeter_identifier in smartmeter_identifiers:
                dtype[f'{smartmeter_identifier}_delta_energy'] = 'float32'

            try:
                new_data = pd.read_json(StringIO(data_json), orient=JSON_ORIENT, dtype=dtype)
            except ValueError as e:
                return jsonify({'message': 'Error in json'}), 400
            
            # check new data
            if new_data.isna().any().any():
                return jsonify({'message': 'Dataframe contains missing values'}), 400
            
            # check timestamp data
            if new_data.timestamp.dtype != 'object':
                return jsonify({'message': f'{new_data.timestamp.dtype} wrong dtype for timestamp column'}), 400
            
            # check delta energy values
            for smartmeter_identifier in smartmeter_identifiers:
                faulty_dtype_columns = []
                column_name = f'{smartmeter_identifier}_delta_energy'
                try:
                    if new_data[column_name].dtype != 'float32':
                        faulty_dtype_columns.append(column_name)
                except KeyError:
                    return jsonify({'message': f'Column {column_name} is not in posted data.'}), 400
                if len(faulty_dtype_columns) > 0:
                    return jsonify({'message': f'Wrong dtype for the columns: {faulty_dtype_columns}'}), 400
                
            # update min and max values
            for smartmeter_identifier in smartmeter_identifiers:
                column_name = f'{smartmeter_identifier}_delta_energy'
                new_min, new_max = new_data[column_name].min().item(), new_data[column_name].max().item()

                # update meta data storage
                self.meta_data[building_identifier].min_values[smartmeter_identifier] = min(self.meta_data[building_identifier].min_values[smartmeter_identifier], new_min)
                self.meta_data[building_identifier].max_values[smartmeter_identifier] = max(self.meta_data[building_identifier].max_values[smartmeter_identifier], new_max)
                
            # convert timestamps to datetime objects
            new_data['timestamp'] = new_data.timestamp.apply(lambda x: datetime.fromisoformat(x))

            if new_data.timestamp.dt.tz != timezone.utc:
                return jsonify({'message': 'Timestamps have wrong timezone'}), 400
                    
            # check if timestamps are consecutive
            new_data['time_diff'] = new_data['timestamp'].diff()
            continuous = (new_data['time_diff'][1:] == pd.Timedelta(minutes=15)).all()

            if not continuous:
                return jsonify({'message': 'Timestamps of new data are not continuous!'}), 400
            else:
                del new_data['time_diff']
            del continuous

            # write new values to database
            fp = self.data_path + building_identifier + ".feather"
            data = pd.read_feather(fp)

            # overwrite existing, add new
            data = pd.merge(left=data, right=new_data, left_on="timestamp", right_on="timestamp", how="outer")
            for smartmeter_identifier in smartmeter_identifiers:
                data.loc[~data[f'{smartmeter_identifier}_delta_energy_y'].isna(), f'{smartmeter_identifier}_delta_energy_x'] = \
                    data.loc[~data[f'{smartmeter_identifier}_delta_energy_y'].isna(), f'{smartmeter_identifier}_delta_energy_y']
                data = data.drop(columns=[f'{smartmeter_identifier}_delta_energy_y'])
                data = data.rename(columns={f'{smartmeter_identifier}_delta_energy_x': f'{smartmeter_identifier}_delta_energy'})

            # ensure sorting
            data.sort_values(by='timestamp', ascending=True, inplace=True)

            # ensure timestamps are continuous
            data['time_diff'] = data['timestamp'].diff()
            continuous = (data['time_diff'][1:] == pd.Timedelta(minutes=15)).all()
            
            if not continuous:
                return jsonify({'message': 'Timestamps of merged data are not continuous'}), 400
            else:
                del data['time_diff']
            del continuous

            data.to_feather(fp)
            
            return jsonify({'message': 'Wrote measurements to database'})

    def check_identifier_dtype(self, identifier:str):
        """
        Checks dtype of identifier and if string starts with correct prefix.

        Params:
        -------
        identifier : str
            Identifier of building or Smartmeter.
        
        Returns:
        --------
        valid : bool
            True if valid dtype and prefix was used. 
        message : str
            Reason for valid.
        """
        if isinstance(identifier, str):
            if identifier.startswith("Building") or identifier.startswith("Smartmeter"):
                return True, f"Identifier {identifier} is valid."
            else:
                return False, f"Identifier {identifier} starts with wrong prefix."
        else:
            return False, f"Dtype {type(identifier)} is invalid dtype for Identifier {identifier}."
    
    def check_smartmeter_identifiers_dtype(self, smartmeter_identifiers:list):
        """
        Checks dtype of smartmeter identifiers.
        
        Params:
        -------
        smartmeter_identifiers : list
            List containing possible smartmeter identifiers.
        
        Returns:
        --------
        valid : bool
            If all identifiers are valid or not.
        message : str
            Why identifiers are valid or not.
        """
        for smartmeter_identifier in smartmeter_identifiers:
            valid, message = self.check_identifier_dtype(identifier=smartmeter_identifier)
            if not valid:
                return valid, message
        return valid, "All dtypes correct"

    def check_building_identifier_db(self, building_identifier:str):
        """
        Checks dtype of building_identifier and if building_identifier is in db.

        Params:
        -------
        building_identifier : str
            Building identifier to check

        Returns:
        --------
        valid : bool
            Boolean describing if building_identifier iis in db or not.
        message : str
            Reason for valid.
        """
        valid, message = self.check_identifier_dtype(identifier=building_identifier)
        if valid:
            if building_identifier in self.meta_data.keys():
                return True, "Building identifier is in database."
            else:
                return False, "Building identifier does not exist in database."
        else:
            return valid, message

    def check_smartmeter_identifiers_db(self, building_identifier:str, smartmeter_identifiers:list):
        """
        Checks dtype of building identifier and of all smartmeter identifiers aswell as if they exist in the database or not.

        Params:
        -------
        building_identifier : str
            Possible building identifier.
        
        smartmeter_identifiers : list
            Possible smartmeter identifiers. 
        
        Returns:
        --------
        valid : bool
            Boolean describing if used parameters where valid or not.
        message : str
            Reason for valid.
        """
        valid, message = self.check_building_identifier_db(building_identifier=building_identifier)

        if valid:
            valid, message = self.check_smartmeter_identifiers_dtype(smartmeter_identifiers=smartmeter_identifiers)                
            if valid:
                ids_in_db, ids_not_in_db = [], []
                for smartmeter_identifier in smartmeter_identifiers:
                    if smartmeter_identifier not in self.meta_data[building_identifier].smartmeter_ids:
                        ids_not_in_db.append(smartmeter_identifier)
                    else:
                        ids_in_db.append(smartmeter_identifier)
                    
                if len(ids_not_in_db) != 0:
                    return False, {
                        'ids_in_db': ids_in_db,
                        'ids_not_in_db': ids_not_in_db
                    }
                
                return True, "Smartmeter identifiers are in database."
            else:
                return valid, message
        else:
            return valid, message
        
    def reload_identifiers(self):
        """Reloads identifiers from disk."""
        # load smartmeter info
        if os.path.exists(self.smartmeter_path):
            with open(self.smartmeter_path, "rb") as f:
                self.meta_data = pickle.load(file=f)
        else:
            self.meta_data = {}

    def store_identifiers(self):
        """Stores identifiers to disk."""
        with open(self.smartmeter_path, "wb") as f:
            pickle.dump(obj=self.meta_data, file=f)