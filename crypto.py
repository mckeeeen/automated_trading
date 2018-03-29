# coding: utf-8

from mckee.crypto_watch import CryptoWatch
import datetime

req = CryptoWatch()
req.get_ohlc()
req.save_ohlc()

print('CryptoWatch Update Finished : ' + str(req.now))