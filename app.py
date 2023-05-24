from flask import Flask, jsonify
from flask_restful import Api, Resource, reqparse
from scripts.detection.train import train_api
from scripts.detection.eval import eval
from scripts.detection.unit import write_to_log


app = Flask(__name__)
api = Api(app)


class ActiveLearning(Resource):
    @staticmethod
    def get():
        write_to_log('start al')
        # номер гпу, 0 или 1...
        device = reqparse.request.args['gpu']
        # сколько семплов вернуть для разметки
        add = int(reqparse.request.args['add'])
        # если датасет большой, то ранжироваться будут batch_unlabeled случайных картинок. если равно -1, то ранжируются все. нор это может быть долго
        batch_unlabeled = int(reqparse.request.args['batch_unlabeled'])

        # путь до неразмеченных фоток и размеченных для обучения
        path_to_labels_train = reqparse.request.args['path_to_labels_train']
        path_to_img_train = reqparse.request.args['path_to_img_train']

        # путь до валидационной части фоток. используется для ранней остановки во время обучения сети
        path_to_labels_val = reqparse.request.args['path_to_labels_val']
        path_to_img_val = reqparse.request.args['path_to_img_val']

        # можно указать путь к весам сети и она будет загружена
        path_model = reqparse.request.args['path_model']

        # если будет обучать с нуля, можно указать что с хаба закачать коковскую сеть
        pretrain = reqparse.request.args['pretrain_from_hub']

        # сеть обученная или загруженная от пользователя будет сохранена со случайным именем
        save_model = reqparse.request.args['save_model']

        # во время обучения будет использоваться ранняя остановка или не будет
        use_val_test = reqparse.request.args['use_val_test_in_train']


        pretrain = True if pretrain == 'T' else False
        save_model = True if save_model == 'T' else False
        # path_model = None if path_model == '' else path_model
        use_val_test = True if use_val_test == 'T' else False

        # сеть указанную пользователем можно переобучить на трейновских данных, или нет
        retrain = reqparse.request.args['retrain_user_model']
        retrain = True if retrain == 'T' else False

        # как обрабатывать несколько боксов на одной фотке.можно выбрать самую низкую уверенность min, самую высокую
        # max или средню по фотке mean
        selection_function = reqparse.request.args['bbox_selection_policy']

        # в каком диапазоне уверенностей отбирать фотки.иногда нужно выбирать самые неуверенные, иногда
        # самые уверенные
        quantile_min = float(reqparse.request.args['quantile_min'])
        quantile_max = float(reqparse.request.args['quantile_max'])

        # сеть для детекции, может быть одной из:
        # yolo, fasterrcnn
        type_model = reqparse.request.args['type_model']

        return train_api(path_to_img_train, path_to_labels_train,
                    path_to_img_val, path_to_labels_val,
                    add, device, path_model, batch_unlabeled, pretrain,
                    save_model, use_val_test, retrain, selection_function,
                    quantile_min, quantile_max, type_model)


class Evaluate(Resource):
    @staticmethod
    def get():
        write_to_log('start mape')
        device = int(reqparse.request.args['gpu'])
        path_to_labels_train = reqparse.request.args['path_to_labels_train']
        path_to_img_train = reqparse.request.args['path_to_img_train']
        path_to_labels_val = reqparse.request.args['path_to_labels_val']
        path_to_img_val = reqparse.request.args['path_to_img_val']

        # точность сети будет считаться на этой отложенной выборки
        path_to_labels_test = reqparse.request.args['path_to_labels_test']
        path_to_img_test = reqparse.request.args['path_to_img_test']

        pretrain = reqparse.request.args['pretrain_from_hub']
        save_model = reqparse.request.args['save_model']

        pretrain = True if pretrain == 'T' else False
        save_model = True if save_model == 'T' else False

        path_model = reqparse.request.args['path_model']
        retrain = reqparse.request.args['retrain_user_model']
        retrain = True if retrain == 'T' else False

        type_model = reqparse.request.args['type_model']

        return eval(path_to_img_train, path_to_labels_train,
                path_to_img_val, path_to_labels_val,
                path_to_img_test, path_to_labels_test, device, save_model,
                pretrain, path_model, retrain, type_model)





class CustomModel(Resource):
    @staticmethod
    def get():
        pass

api.add_resource(ActiveLearning, '/active_learning')
api.add_resource(Evaluate, '/eval')

if __name__ == '__main__':
    app.run(host='127.0.0.1', debug=True)
