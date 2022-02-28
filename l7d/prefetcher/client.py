import blosc2
import dill
import io
from kafka import KafkaConsumer
from kafka import KafkaProducer
from multiprocessing import Queue
import numpy as np
from PIL import Image
import sys
import time
import traceback
from threading import Thread


class Consumer(Thread):
    def __init__(self, clientId, addressList, maxByteSize, queue):
        super().__init__()
        self.__clientId = clientId
        self.__addressList = addressList
        self.__maxByteSize = maxByteSize
        self.__dataQueue = queue

    def parseHeaders(self, headers: tuple) -> dict:
        headers = {
            'id': headers[0][1].decode('utf-8'),
            'meta': headers[1][1].decode('utf-8'),
            'time': headers[2][1].decode('utf-8'),
            'start time': headers[3][1].decode('utf-8'),
        }
        return headers

    def parseRecord(self, record):
        headers = self.parseHeaders(record.headers)
        data = blosc2.decompress(record.value)
        return data, headers

    def run(self):
        consumer = KafkaConsumer(
            self.__clientId,
            bootstrap_servers=self.__addressList,
            fetch_max_bytes=self.__maxByteSize
        )

        while True:
            for record in consumer:
                data, headers = self.parseRecord(record)
                self.__dataQueue.put((data, headers['time'], time.time() - float(headers['start time'])))


class Prefetcher(Thread):
    def __init__(
            self,
            addressList: list=None,
            clientId: str=None,
            batchSize: int=16,
            dataset: list=None,
            meta: str=None,
            module: type=None,
    ):
        super().__init__()
        self.daemon = True
        self.__addressList = addressList
        self.__clientId = self.setClientId(clientId)
        self.__batchSize = self.setBatchSize(batchSize)
        self.__dataset = self.setDataset(dataset)
        self.__meta = self.setMetadata(meta)
        self.__module = self.setModule(module)
        self.__lastIndex = 0
        self.__maxByteSize = 10485760
        self.__headers = self.setHeaders()
        self.__dataQueue = Queue()

    def setClientId(self, clientId):
        if isinstance(clientId, str):
            return clientId
        else:
            raise ValueError('"clientId" must be string.')

    def setBatchSize(self, batchSize):
        if isinstance(batchSize, int):
            if batchSize < 1:
                raise ValueError('"batchSize" must be greater than 0.')
            return batchSize
        else:
            raise ValueError('"batchsize" must be int.')

    def setDataset(self, dataset):
        if isinstance(dataset, list):
            if dataset:
                return dataset
            else:
                raise ValueError('"data" should not be None.')
        else:
            raise ValueError('"data" must be list.')

    def setMetadata(self, meta):
        if isinstance(meta, str):
            if meta == 'image':
                return 'image'
            elif meta == 'ndarray':
                return 'ndarray'
            else:
                raise ValueError('"meta" accepts either "image" or "ndarray".')
        else:
            raise ValueError('"meta" must be string.')

    def setModule(self, module):
        if isinstance(module, type):
            return module
        else:
            raise ValueError('module')

    def setHeaders(self):
        headers = [
            ('id', self.__clientId.encode('utf-8')),
            ('meta', self.__meta.encode('utf-8')),
            ('module', dill.dumps(self.__module)),
            ('start time', str(time.time()).encode('utf-8'))
        ]
        return headers

    def getQueue(self):
        return self.__dataQueue

    def readFile(self, dataPath):
        with open(dataPath, 'rb') as dataFile:
            file = dataFile.read()
        return file

    def sendData(self, producer, data):
        if self.__meta == 'image':
            fileBytes = self.readFile(data)
            producer.send(
                'preprocess',
                value=fileBytes,
                headers=self.__headers
            )

        elif self.__meta == 'ndarray':
            dataArray = self.image2Array(data)
            producer.send(
                'preprocess',
                value=dataArray,
                headers=self.__headers
            )
            print('sent')

    def image2Array(self, filePath):
        dataBytes = self.readFile(filePath)
        dataArray = np.array(Image.open(io.BytesIO(dataBytes))).tobytes()
        dataArray = blosc2.compress(dataArray)
        print('raw : ', sys.getsizeof(dataBytes))
        print('compressed : ', sys.getsizeof(dataArray))
        return dataArray

    def run(self):
        self.consumer = Consumer(self.__clientId, self.__addressList, self.__maxByteSize, self.__dataQueue)
        self.consumer.daemon = True
        self.consumer.start()
        producer = KafkaProducer(
            bootstrap_servers=self.__addressList,
            max_request_size=self.__maxByteSize
        )
        dataLength = len(self.__dataset)

        while self.__lastIndex < dataLength:
            try:
                end = self.__lastIndex + self.__batchSize
                if end < dataLength:
                    for index in range(self.__lastIndex, end):
                        self.sendData(producer, self.__dataset[index])
                else:
                    for index in range(self.__lastIndex, dataLength):
                        self.sendData(producer, self.__dataset[index])
                self.__lastIndex += self.__batchSize
            except:
                print(traceback.print_exc())
                break
        print('done')
