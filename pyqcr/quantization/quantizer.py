class ModelQuantizer:

    def __init__(self, quant_spec, precision):
        self.quant_spec = quant_spec
        self.precision = precision

    def quantize(self, model):
        return model

    def calibrate(self, model_fakeq, calib_data_func, method):
        return model_fakeq

    def convert_to_int(self, model_fakeq):
        return model_fakeq
