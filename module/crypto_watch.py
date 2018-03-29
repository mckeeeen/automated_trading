import pandas as pd
import datetime
import requests


class CryptoWatch(object):
    '''
    arguments
    
    exchange: [str]
    pair: [str]
    before: [int/unix timestamp]
    after: [int/unix timestamp]
    span: [int/days]
    periods: [list/int,int,.../seconds]
    
    '''
    
    def __init__(self, exchange='bitflyer',pair='btcjpy', before=None, after=None, span=30, periods=[60,180,300,900,1800,3600,7200,14400,21600,43200,86400,259200,604800]):
        
        self.exchange = exchange
        self.pair = pair
        self.now = datetime.datetime.now()
        self.span = span
        self.periods = [str(i) for i in periods]
        self.periods_ = ','.join(self.periods)
        
        if before:
            self.before = before
        else:
            self.before = self.now.strftime('%s')
        
        if after:
            self.after = after
        else:
            self.after = (self.now - datetime.timedelta(days=self.span)).strftime('%s')
        
        self.ohlc = []
        self.params_ohlc = 0
        self.url_ohlc = 0
        self.request_ohlc = 0
            
    
    def get_ohlc(self):
        
        self.params_ohlc = {'before':self.before,'after':self.after,'periods':self.periods_}
        self.url_ohlc = 'https://api.cryptowat.ch/markets%s%s/ohlc'%('/'+self.exchange,'/'+self.pair)
        self.request_ohlc = requests.get(self.url_ohlc, params=self.params_ohlc)
        for i in range(len(self.periods)):
            df = pd.DataFrame(self.request_ohlc.json()['result'][self.periods[i]], columns=['CloseTime','OpenPrice','HighPrice','LowPrice','ClosePrice','Volume[BTC]','Volume[JPY]'])
            df['CloseTime[UTC]'] = pd.to_datetime(df['CloseTime'], unit='s')
            df['CloseTime[JST]'] = df['CloseTime[UTC]'] + pd.Timedelta(9,unit='h')
            self.ohlc.append(df)
            df = None
            
    def save_ohlc(self, directory='./data'):
        
        for i in range(len(self.periods)):
            path = '%s/%s_%s_%s_seconds.json'%(directory, self.exchange,self.pair,self.periods[i])

            try:                
                j = pd.read_json(path)
                j.index = j.index.astype(int)
                j.sort_index(inplace=True)
                last_unix = j['CloseTime'].iloc[-1]
                last_index = self.ohlc[i][self.ohlc[i].CloseTime == last_unix].index[0]
                new = self.ohlc[i].iloc[last_index+1:,:]
                j = pd.concat([j,new],ignore_index=True)

            except ValueError:
                j = self.ohlc[i]

            j.to_json(path)
            
            path = 0
            last_unix = 0
            lase_index = 0