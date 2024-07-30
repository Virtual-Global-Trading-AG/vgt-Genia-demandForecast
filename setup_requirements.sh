#!/bin/bash

pip freeze > requirements.txt
cp requirements.txt demandForecastingManager/requirements.txt
cp requirements.txt demandForecastingService/requirements.txt