import glob
import cv2 as cv
import numpy as np
from time import time
from openvino.inference_engine import IENetwork, IEPlugin

TEMP = None

class Detector:
    def __init__(self, device, model_xml, model_bin):
        """
            Supported devices: 
                CPU, 
                GPU, 
                FPGA, 
                MYRIAD (Intel Neural Compute Stick 2), 
                HETERO, 
                MULTI
        """
        checkpt = time()
        self.plugin = IEPlugin(
            device=device, plugin_dirs='/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/')
        self.network = IENetwork(model=model_xml, weights=model_bin)

        print('out:', len(self.network.outputs))
        print('outputs', self.network.outputs)

        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))

        self.network.batch_size = 1
        
        self.input_shape = self.network.inputs[self.input_blob].shape
        self.n, self.channel, self.h, self.w = self.input_shape
        print('Input shape: {}'.format(self.input_shape))
        self.exec_net = self.plugin.load(network=self.network)
        print("=== Finish Detector Init ({:.2f} sec)===".format(time()-checkpt))

    def load_image(self, path):
        self.image = cv.imread(path)
        if self.image.shape[:-1] != (self.h, self.w):
            image_resized = cv.resize(self.image, (self.w, self.h))
        image_openvino = image_resized.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        self.image_openvino = np.expand_dims(image_openvino, axis=0)

    def process(self):
        result = self.exec_net.infer(inputs={self.input_blob: self.image_openvino})
        self.result_output = result[self.output_blob]


    def get_not_support_layer(self):
        supported_layers = plugin.get_supported_layers(self.network)
        not_supported_layers = [
            l for l in self.network.layers.keys() if l not in supported_layers]
        print('not supported layers:', not_supported_layers)

    def get_output(self, min_score=0):
        global TEMP
        """
            openvino format is [N/A, label, score, xmin, ymin, xmax, ymax]
        """
        detections_filtered = [
            i for i in self.result_output[0][0] if i[2] > min_score]
        return detections_filtered

def display_output(image, result, definition, path, output_method='show'):
    for i in range(len(result)):
        shape = image.shape
        label = result[i][1]
        score = result[i][2]
        boxes = result[i][3:]
        x_min = int(boxes[0] * shape[1])
        y_min = int(boxes[1] * shape[0])
        x_max = int(boxes[2] * shape[1])
        y_max = int(boxes[3] * shape[0])

        color = definition[label]['color']
        cv.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
        font = cv.FONT_HERSHEY_SIMPLEX
        text = definition[label]['label'] + ': {:.2f}'.format(score) 
        cv.putText(image,text ,(x_min,y_min), font, 1,color,2,cv.LINE_AA)
    
    if output_method == 'show':
        while True:
            cv.imshow('image', image)
            if cv.waitKey(1) & 0xFF == 27:
                break
    elif output_method == 'write':
        cv.imwrite('out-img/' + path.split('/')[-1], image)

    


def main():
    definition = {
        0:{
            'label': 'gate',
            'color': (0, 255 ,0)
        },
        1: {
            'label': 'flare',
            'color': (0,0,255)
        }
    }

    image_paths = glob.glob('/home/jaogon/zbus_ml/demo code/imgs/*png')[:50]
    model_xml = '/home/jaogon/zbus_ml/demo code/resnet50.xml'
    model_bin = '/home/jaogon/zbus_ml/demo code/resnet50.bin'
    # device = "CPU"
    device = "MYRIAD"
    # device = "GPU"
    detector = Detector(device=device, model_xml=model_xml, model_bin=model_bin)

    service_time = []
    for path in image_paths:
        checkpt = time()
        detector.load_image(path)
        detector.process()
        result = detector.get_output(min_score=0.1)
        service_time.append(time()-checkpt)
        display_output(detector.image, result, definition, path, output_method='write')
        print('{}/{}'.format(len(service_time),len(image_paths)))

    print('Max: {}\nMin: {}\nAvg: {}'.format(max(service_time), min(service_time), sum(service_time)/len(service_time)))


if __name__ == '__main__':
    main()
