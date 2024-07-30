import pandas as pd
import numpy as np
import pytz
from flask import Flask, jsonify, request
from datetime import datetime, timedelta
import os
import requests
import pickle

JSON_ORIENT = "split"

class demandForecastManager:
    """
    Manages energy consumption data from Genia instances. 

    Tasks:
        - Accept and store new measurement data
        - Trigger training of new demand forecast models per instance
        - Return genia instance specific forecasts
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
        self.smartmeter_identifiers = {} # dictionary 
        self.reload_identifiers()

    def configure_routes(self):
        """Sets up flask routes."""
        @self.app.route("/", methods=['GET'])
        def index():
            return jsonify({'message': 'Index of demandForecastManager.'}), 200
        
        @self.app.route("/getHealth", methods=['GET'])
        def get_health():
            return jsonify({'status':'up', 'lastRestart': self.last_restart.isoformat()}), 200
        
        @self.app.route("/getIds", methods=['GET'])
        def get_ids():
            """Return all building identifiers in database."""
            return jsonify({'buildingIds': self.smartmeter_identifiers})

        ## add new id
        @self.app.route("/addId", methods=['POST'])
        def add_id():
            # parse and check identifiers from request
            building_identifier = request.headers.get("buildingIdentifier")
            smartmeter_identifiers = request.headers.get("smartmeterIdentifiers")
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
            self.smartmeter_identifiers[building_identifier] = smartmeter_identifiers

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
            self.smartmeter_identifiers[building_identifier].extend(ids_not_in_db)

            # add new columns to database
            try:
                data = pd.read_feather(f"{self.data_path}/{building_identifier}.feather")
            except FileNotFoundError:
                return jsonify({'message': 'Missing database entry for buildingIdentifier.'}), 400
            
            for smartmeter_identifier in ids_not_in_db:
                data[smartmeter_identifier + "_delta_energy"] = np.nan
            
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
                self.smartmeter_identifiers[building_identifier].remove(smartmeter_identifier)

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
            del self.smartmeter_identifiers[building_identifier]

            # delete dataframe
            os.remove(f"{self.data_path}{building_identifier}.feather" )

            return jsonify({'message':'Building identifier successfully removed from database.'}), 200
        
        @self.app.route("/getMeasurements", methods=['GET'])
        def get_measurements():
            building_identifier = request.header.get("buildingIdentifier")
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
            if building_identifier not in self.smartmeter_identifiers.keys():
                return jsonify({'message': f'Building identifier {building_identifier} does not exist in database.'}), 404
            
            data = pd.read_feather(f"{self.data_path}{building_identifier}.feather")
            data = data[(data.timestamp >= start) & (data.timestamp <= end)]
            data_json = data.to_json(orient=JSON_ORIENT)

            return jsonify({'data': data_json}), 200
        
        @self.app.route("/postMeasurements", methods=['POST'])
        def post_measurements():
            building_identifier = request.headers.get('buildingIdentifier')
            smartmeter_identifiers = request.headers.get('smartmeterIdentifiers')
            data_json = request.json

            # check building identifiers and smartmeter identifiers
            valid, message = self.check_smartmeter_identifiers_db(building_identifier=building_identifier, smartmeter_identifiers=smartmeter_identifiers)
            if not valid:
                return jsonify({'message': message}), 400
            
            if data_json is None:
                return jsonify({'message': "Missing data."}), 400
            
            # convert to dataframe
            try:
                new_data = pd.DataFrame(data_json)
            except ValueError:
                return jsonify({'message': 'Error in json'}), 400
            
            # check new data
            if new_data.isna().any().any():
                return jsonify({'message': 'Dataframe contains missing values'}), 400
            
            # write new values to database
            fp = self.data_path + building_identifier + ".feather"
            data = pd.read_feather(fp)
            data.update(new_data)
            data = new_data.combine_first(data)
            data.to_feather(fp)
            
            return jsonify({'message': 'Wrote measurements to database'})
        
        @self.app.route("/getForecast", methods=['GET'])
        def get_forecast():
            building_identifier = request.headers.get("buildingIdentifier")
            valid, message = self.check_building_identifier_db(building_identifier=building_identifier)
            
            if not valid:
                return jsonify({'message': message}), 400
            
            # TODO make request to forecasting service
            raise NotImplementedError

            # if forecasting service is not available
            return jsonify({'message': 'Forecasting service is down. '}), 503

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
            if building_identifier in self.smartmeter_identifiers.keys():
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
                    if smartmeter_identifier not in self.smartmeter_identifiers[building_identifier]:
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
                self.smartmeter_identifiers = pickle.load(file=f)
        else:
            self.smartmeter_identifiers = {}

    def store_identifiers(self):
        """Stores identifiers to disk."""
        with open(self.smartmeter_path, "wb") as f:
            pickle.dump(obj=self.smartmeter_identifiers, file=f)