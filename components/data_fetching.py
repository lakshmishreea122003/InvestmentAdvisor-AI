import time as tm
import datetime as dt
from yahoo_fin import stock_info as yf

class Fetch:

    def __init__(self, stock):
        self.date_now = tm.strftime('%Y-%m-%d')
        self.date_3_years_back = (dt.date.today()-dt.timedelta(days=6)).strftime('%Y-%m-%d')
        self.stock = stock

    def fetch_data(self):
        init_data = yf.get_data(self.stock, start_date=self.date_3_years_back, end_date=self.date_now, interval = '1d')
        print(len(init_data))
        return init_data