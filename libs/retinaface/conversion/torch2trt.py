import argparse
import torch
# from data import cfg_mnet, cfg_slim, cfg_rfb
from config import cfg_mnet, cfg_slim, cfg_rfb
# from torch2trt_dynamic import torch2trt_dynamic

# from models.retinaface import RetinaFace
# this version remove landmark head
from models.retinaface_trim import RetinaFace, ProcModel

import onnx
from models.net_slim import Slim
from models.net_rfb import RFB
import numpy as np


parser = argparse.ArgumentParser(description='Test')
parser.add_argument('-m', '--trained_model', default='../../weight/mobilenet0.25_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='mobile0.25',
                    help='Backbone network mobile0.25 or slim or RFB')
# parser.add_argument('--long_side', default=320, help='when origin_size is false, long_side is scaled size(320 or 640 for long side)')
parser.add_argument('--width', type=int, default=320)
parser.add_argument('--height', type=int, default=288)
parser.add_argument('--cpu', action="store_true",
                    default=False, help='Use cpu inference')

args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    def f(x): return x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def create_attrs(input_h, input_w, topK, keepTopK):
    attrs = {}
    attrs["shareLocation"] = 1
    attrs["backgroundLabelId"] = -1
    attrs["numClasses"] = 1
    attrs["topK"] = topK
    attrs["keepTopK"] = keepTopK
    attrs["scoreThreshold"] = 0.4
    attrs["iouThreshold"] = 0.6
    attrs["isNormalized"] = False
    attrs["clipBoxes"] = False
    # 001 is the default plugin version the parser will search for, and therefore can be omit
    # but we include it here for illustrative purposes.
    attrs["plugin_version"] = "1"

    return attrs


def export_onnx(model, im, file, opset=12, dynamic=True, nmsplugin_included=False):
    # YOLOv5 ONNX export

    output_names = ['output0-bbox', 'output1-class']
    if dynamic:
        file = file.split('.')[0] + '_dynamic'
    if nmsplugin_included:
        file = file.split('.')[0] + '_nms'
    f = file + '.onnx'
    if nmsplugin_included:
        output_names = ['output0-bbox', 'output1-class']
    if dynamic:
        # dynamic = {'images': {0: 'batch', 2: 'height', 3: 'width'}}  # shape(1,3,640,640)
        dynamic = {'images': {0: 'batch'}}
        dynamic['output0-bbox'] = {0: 'batch'}
        dynamic['output1-class'] = {0: 'batch'}

    torch.onnx.export(
        model.cpu() if dynamic else model,  # --dynamic only compatible with cpu
        im.cpu() if dynamic else im,
        f,
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,  # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
        input_names=['images'],
        output_names=output_names,
        dynamic_axes=dynamic or None)
    if nmsplugin_included:
        add_nmsplugin_to_onnx(
            model_file=f
        )

    # Checks
    # model_onnx = onnx.load(f)  # load onnx model
    # onnx.checker.check_model(model_onnx)  # check onnx model

    # # Metadata
    # d = {'stride': int(max(model.stride)), 'names': model.names}
    # for k, v in d.items():
    #     meta = model_onnx.metadata_props.add()
    #     meta.key, meta.value = k, str(v)
    # onnx.save(model_onnx, f)
    # Simplify
    return f


def add_nmsplugin_to_onnx(model_file, output_names=('output0-bbox', 'output1-class'), topk=100, keepTopK=1):
    import onnx_graphsurgeon as gs

    graph = gs.import_onnx(onnx.load(model_file))  # load onnx model
    batch_size = graph.inputs[0].shape[0]
    input_h = graph.inputs[0].shape[2]
    input_w = graph.inputs[0].shape[3]
    tensors = graph.tensors()
    boxes_tensor = tensors[output_names[0]]  # match with onnx model output name
    confs_tensor = tensors[output_names[1]]  # match with onnx model output name

    num_detections = gs.Variable(name="num_detections").to_variable(dtype=np.int32, shape=[batch_size, 1])
    nmsed_boxes = gs.Variable(name="nmsed_boxes").to_variable(dtype=np.float32, shape=[batch_size, keepTopK, 4])
    # nmsed_scores = gs.Variable(name="nmsed_scores").to_variable(dtype=np.float32, shape=[batch_size, keepTopK])
    # nmsed_classes = gs.Variable(name="nmsed_classes").to_variable(dtype=np.float32, shape=[batch_size, keepTopK])

    new_outputs = [num_detections, nmsed_boxes]  #  , nmsed_scores, nmsed_classes] # do not change

    nms_node = gs.Node(  # define nms plugin
        op="BatchedNMSDynamic_TRT",  # match with batchedNMSPlugn
        attrs=create_attrs(input_h, input_w, topk, keepTopK),  # set attributes for nms plugin
        inputs=[boxes_tensor, confs_tensor],
        outputs=new_outputs
    )
    graph.nodes.append(nms_node)  # nms plugin added
    graph.outputs = new_outputs
    graph = graph.cleanup().toposort()

    onnx.save(gs.export_onnx(graph), model_file)  # save model
    return model_file


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(
            pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(
            pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


if __name__ == '__main__':
    torch.set_grad_enabled(False)

    cfg = None
    net = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
        net = RetinaFace(cfg=cfg, phase='test')
    elif args.network == "slim":
        cfg = cfg_slim
        net = Slim(cfg=cfg, phase='test')
    elif args.network == "RFB":
        cfg = cfg_rfb
        net = RFB(cfg=cfg, phase='test')
    else:
        print("Don't support network!")
        exit(0)

    # load weight
    net = load_model(net, args.trained_model, args.cpu)
    print('Finished loading model!')
    device = torch.device("cpu" if args.cpu else "cuda:1")

    net = ProcModel(net, class_num=2, w=320, h=288, device='cpu')
    net = net.to('cpu')

    # export
    inputs = torch.randn(1, 3, args.height, args.width)
    export_onnx(net, inputs, 'retinacroppedface.onnx', nmsplugin_included=True)
    # print(outputs[0].shape, outputs[1].shape)
    # input_names = ["input_det"]
    # output_names = ["output_det0", "output_det1"]
    # # convert to TensorRT feeding sample data as input
    # opt_shape_param = [
    #     [
    #         [1, 3, args.height, args.width],   # min
    #         [1, 3, args.height, args.width],   # opt
    #         [1, 3, args.height, args.width]    # max
    #     ]
    # ]
    # print('torch2trt_dynamic')
    # model_trt = torch2trt_dynamic(
    #     net, [inputs], fp16_mode=True, opt_shape_param=opt_shape_param, input_names=input_names, output_names=output_names)
    # save_path = f'retina-{args.network}-{args.height}x{args.width}-b1-fp16.engine'
    # print('Saving')
    # with open(save_path, 'wb') as f:
    #     f.write(model_trt.engine.serialize())
