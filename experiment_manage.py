import os
import time, csv
import make_args
import itertools


class Recorder:
    def __init__(self):
        pass

    def step(self):
        pass

class ParameterRecorder(Recorder):
    def __init__(self, tobe_adjust_parameters_dir):
        super(ParameterRecorder, self).__init__()
        self.tobe_adjust_parameters_dir = tobe_adjust_parameters_dir
        self.list_len = {}
        self.parameter_idx_mark = 0
        self.adjust_finished_mark = {}
        self.eof = False
        for k in tobe_adjust_parameters_dir.keys():
            self.list_len[k] = tobe_adjust_parameters_dir[k].__len__() - 1
            self.adjust_finished_mark[k] = False
        self.construct()

    def construct(self):
        self.opt = make_args.args()
        self.group = self.set_group()

    def set_group(self):
        arg_num = len(self.tobe_adjust_parameters_dir)
        keys = list(self.tobe_adjust_parameters_dir.keys())
        keys.reverse()
        root = [[v] for v in self.tobe_adjust_parameters_dir[keys[0]]]
        for i in range(1, len(keys)):
            k = keys[i]
            now_vertex = self.tobe_adjust_parameters_dir[k]
            tmp_root = []
            tmp_r = root.copy()
            for arg in now_vertex:
                for r in tmp_r:
                    t_r = r.copy()
                    t_r.insert(0, arg)
                    tmp_root.append(t_r)
            root = tmp_root
        return root

    def detect_eof(self):
        if self.parameter_idx_mark < self.group.__len__() - 1:
            return False
        return True

    def get_status(self):
        # print(self.parameter_idx_mark)
        now_status = {k: self.group[self.parameter_idx_mark][i] for i, k in enumerate(self.tobe_adjust_parameters_dir.keys())}

        return now_status, self.detect_eof()

    def update_status(self):
        idx = self.parameter_idx_mark
        paras = self.group[idx]
        for i, k in enumerate(self.list_len.keys()):
                self.opt.__dict__[k] = paras[i]
        if self.parameter_idx_mark < self.group.__len__() - 1:
            self.parameter_idx_mark += 1

    def step(self):
        self.status, eof = self.get_status()
        self.update_status()
        return self.opt, self.status, eof





import  time, datetime
class AccFileWriter:
    def __init__(self, dir, parameters_dir):
        self.dir = dir
        if not os.path.exists(dir):
            os.mkdir(dir)
        # self.date = time.strftime("%Y-%m-%d-%h-", time.localtime())
        self.date = str(datetime.datetime.now())
        self.transfer = '%s2%s' %(parameters_dir['source_domain'], parameters_dir['target_domain'])
        self.record_path = os.path.join(dir, self.date+self.transfer+'-'+parameters_dir['metric'][0]+'-'+parameters_dir['data_name'][0], )
        if not os.path.exists(self.record_path):
            os.mkdir(self.record_path)
        time_now = time.asctime(time.localtime(time.time()))
        # self.clear_empty_history()
        self.file_path = os.path.join(self.record_path, 'results-(%s).csv'%str(time_now))
        self.keys = list(parameters_dir.keys())
        self.keys.append('acc')
        self.keys.append('val_acc')
        with open(self.file_path, 'w', encoding='utf-8') as res_file:
            writer = csv.writer(res_file)
            writer.writerow(self.keys)

    def clear_empty_history(self):
        print('clearing the history in the \'%s\'...' % self.file_path)
        for f in os.listdir(self.file_path):
            if '.csv' in f:
                csv_f = os.path.join(self.file_path, f)
                with open(csv_f, 'r', encoding='utf-8') as file:
                    reader = csv.reader(file)
                    if reader.__len__() <= 1:
                        os.remove(csv_f)

    def update(self, results, now_parmeters_dir):
        acc_, h_ = results['aver_accuracy'], results['aver_h']
        acc = '%.3f+-%.2f' % (acc_, h_)
        para = [now_parmeters_dir[k] for k in self.keys if k != 'acc' and k != 'val_acc']

        with open(self.file_path, 'a+', encoding='utf-8') as res_file:
            writer = csv.writer(res_file)
            para.append(acc)
            para.append('%.3f'%results['val_acc'])
            writer.writerow(para)


import shutil
class ExperimentServer:
    def __init__(self, parameters_dir, func_tobe_excuted):

        self.adjust_parameters_dir = parameters_dir
        self.func_tobe_excuted = func_tobe_excuted
        self.file_writer = AccFileWriter(dir='./adjust_parameters', parameters_dir=parameters_dir)
        self.record_path = self.file_writer.record_path
        self.parameter_record = ParameterRecorder(parameters_dir)
        self.construct()
        self.run_mark = False
        self.DA = True

    def construct(self):
        self.opt = self.parameter_record.opt
        self.opt.record_path = self.record_path

    def run(self):
        eof = False
        ablative_list = [
                         [0.99, 0, 0],
                         [0.99, 0.5, 0],
                         [0.99, 0.5, 100]]
        while not eof:
            try:
                print('training...')
                self.opt, now_parmeters_dir, eof = self.parameter_record.step()
                self.opt.outf = self.file_writer.record_path
                # model_path = ['%s:%s'%(str(k), str(now_parmeters_dir[k])) for k in list(now_parmeters_dir.keys())]
                # a = ''
                # for s in model_path:
                #     a+= '_' +s
                a = 'saved_models'
                self.opt.outf = os.path.join(self.opt.outf, a)
                if self.opt.target_domain == self.opt.source_domain and self.opt.DA:
                    print('skip the %s to %s. \n' %(self.opt.source_domain, self.opt.target_domain))
                    continue
                # if [self.opt.loss_weight, self.opt.soft_weight , self.opt.cov_weight] not in ablative_list:
                #     continue
                results = self.func_tobe_excuted(self.opt)
                # step to next parameter, and write the result to .csv file
                self.file_writer.update(results, now_parmeters_dir)
            except BaseException:
                import traceback
                # if isinstance(e, KeyboardInterrupt):
                for s in os.listdir(self.opt.outf):
                    os.remove(os.path.join(self.opt.outf, s))
                with open(self.file_writer.file_path, 'r', encoding='utf-8') as csv_file:
                    readline = csv.reader(csv_file)
                    for i, line in enumerate(readline):
                        if i >= 1:
                            exit()
                    os.remove(self.file_writer.file_path)
                if os.path.exists(self.opt.outf):
                    for p in [os.path.join(self.opt.outf, otf) for otf in os.listdir(self.opt.outf)]:
                        try:
                            os.remove(p)
                        except:
                            print(p)
                    os.removedirs(self.opt.outf)
                for f in os.listdir(self.record_path):
                    for k in os.listdir(os.path.join(self.record_path, f)):
                        try:
                         os.close(open(os.path.join(self.record_path, f, k), "r"))
                         os.remove(os.path.join(self.record_path, f, k))
                        except:
                            print(k)
                    shutil.rmtree(os.path.join(self.record_path, f), ignore_errors=True)
                    # os.removedirs(os.path.join(self.record_path, f))
                shutil.rmtree(self.record_path, ignore_errors=True)

                # os.removedirs(self.record_path)
                # try:
                #     os.removedirs(self.record_path)
                # except:
                #     os.removedirs(self.record_path)
                # if os.listdir(self.file_writer.record_path).__len__() < 1:
                #     os.removedirs(self.file_writer.record_path)
                print(traceback.print_exc())
                exit()







class History:
    def __init__(self, parameters):
        self.parameters = parameters
        self.histor_ = {'param':[], 'acc':[]}

    def store(self, now_status):
        self.histor_ ['param'].append(now_status['paramters'])
        self.histor_ ['acc'].append(now_status['acc'])


    def feed_back(self):
        max_acc = max(self.histor_ ['acc'])
        max_acc_idx = self.histor_['acc'].index(max_acc)
        max_acc_params = self.histor_['param'][max_acc_idx]
        return max_acc_params, max_acc

    def __call__(self, now_status):
        self.store(now_status)
        return self.feed_back()

class Predictor:
    def __init__(self, parameters_dir, strides, range):
        self.parameters_dir = parameters_dir
        self.strides = strides
        self.range = range
        pass

    def oracle(self, max_acc, max_acc_params):
        pass

    def __call__(self, max_acc, max_acc_params, history):
        pass


class ParameterHound:
    def __init__(self, parameter_dir):
        """
        :param parameter_dir: {'parameter_name:' range-->[low bounce, high bounce]}
        """
        self.parameter_dir = parameter_dir
        self.predictor = None
        self.now_status = None
        self.next_status = None
        self.max_acc_params = 0
        self.max_acc = 0
        self.history = History(parameter_dir)

    def search(self):
        max_acc = self.max_acc
        max_acc_params = self.max_acc_params
        prediction = self.predictor(max_acc, max_acc_params)
        self.update_status(prediction)

    def store_history(self, results):
        self.max_acc_params, self.max_acc = self.history({'paramters': self.now_status, 'acc': results})

    def update_status(self, prediction):
        self.now_status = self.next_status
        self.next_status = prediction

    def run(self, results):
        self.store_history(results)
        self.search()
        assert self.next_status != None
        return self.next_status
