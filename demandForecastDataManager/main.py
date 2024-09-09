from demandForecastDataManager.demandDataForecastManager import demandForecastDataManager

dfm = demandForecastDataManager()
app = dfm.app

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5012)