import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, request, jsonify
from datetime import datetime, timedelta, timezone
from xgboost import XGBRegressor

class demandForecastService:
    """
    Manages training and inference of demand forecasting.
    """
    def __init__(self, url_demand_forecast_manager:str):
        assert isinstance(url_demand_forecast_manager, str)
        assert url_demand_forecast_manager.startswith("http://") or url_demand_forecast_manager.startswith("https://")
        self.url_demand_forecast_manager = url_demand_forecast_manager
        self.last_restart = datetime.now()
        self.app = Flask(import_name=__name__)
        self.configure_routes()


        pass

    def configure_routes(self):
        @self.app.route("/", methods=['GET'])
        def index():
            return jsonify({'message': 'Index of Demand Forecast Service'}), 200
        
        @self.app.route("/getHealth", methods=['GET'])
        def get_health():
            return jsonify({'message': 'demandForecastService is up.', 'lastRestart': self.last_restart.isoformat()})
        
        @self.app.route("/getForecast", methods=['GET'])
        def get_forecast():
            """Gets forecast from xgboost model."""
            return jsonify()
        
        @self.app.route("/train", methods=['GET'])
        def train():
            """Trains xgboost model."""
            return jsonify()
        

    def train(self):
        """"""

    def preprocess_data(self):
        """"""
        # scaling

        # feature engineering

        # return


    def load_data(self, start, stop):
        """"""
        # make request to demandforecastmanager

        # preprocessing

        # return