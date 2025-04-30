import sys
sys.path.append('../')
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call

import pyds

CLASS_LABELS = [
    "beer_tiger", "bottle", "can", "cocacola", "cocacola_light", "green_tea", "pepsi", 
    "red_bull", "revive_lemon_salt", "revive_regular", "strawberry_sting", "vinh_hao_water"
]

CLASS_IDS = {label: idx for idx, label in enumerate(CLASS_LABELS)}


def osd_sink_pad_buffer_probe(pad,info,u_data):
    frame_number=0
    num_rects=0

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
           frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        # Intializing object counter with 0
        obj_counter = {class_id: 0 for class_id in CLASS_IDS.values()}

        frame_number=frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta
        l_obj=frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            obj_counter[obj_meta.class_id] += 1
            obj_meta.rect_params.border_color.set(
                0.0, 0.0, 1.0, 0.8
            )
            try: 
                l_obj=l_obj.next
            except StopIteration:
                break

        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]
        
        count_text = f"Frame={frame_number} | Total Objects={num_rects}"
        for class_name, class_id in CLASS_IDS.items():
            count_text += f" | {class_name}: {obj_counter[class_id]}"
        
        py_nvosd_text_params.display_text = count_text

        # Now set the offsets where the string should appear
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12

        # Font , font-color and font-size
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 10
        # set(red, green, blue, alpha); set to White
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

        # Text background color
        py_nvosd_text_params.set_bg_clr = 1
        # set(red, green, blue, alpha); set to Black
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
        # Using pyds.get_string() to get display_text as string
        print(pyds.get_string(py_nvosd_text_params.display_text))
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)
        try:
            l_frame = l_frame.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def main():
    # Standard GStreamer initialization
    Gst.init(None)
    
    # Create Gstreamer elements
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()

    if not pipeline:
        sys.stderr.write("Unable to create Pipeline\n")
        sys.exit(1)

    streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
    streammux.set_property('width', 1920)
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', 3)  # Số lượng camera
    streammux.set_property('batched-push-timeout', 40000)
    streammux.set_property('live-source', 1)

    pipeline.add(streammux)
    
    # Source element for reading from the file
    sources = []
    # camera_devices = ['/dev/video0', '/dev/video2', '/dev/video4']
    camera_devices = ['/dev/video0', '/dev/video4']

    for i, device in enumerate(camera_devices):  
        source = Gst.ElementFactory.make("v4l2src", f"usb-cam-source-{i}")
        source.set_property('device', device)
        capsfilter = Gst.ElementFactory.make("capsfilter", f"v4l2src_caps_{i}")
        capsfilter.set_property('caps', Gst.Caps.from_string("video/x-raw, width=640, height=360, framerate=10/1"))
        # capsfilter.set_property('caps', Gst.Caps.from_string("video/x-raw"))
        vidconv = Gst.ElementFactory.make("videoconvert", f"convertor_src_{i}")
        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", f"convertor_src_nv_{i}")
        nvvidconv.set_property("nvbuf-memory-type", 0)
        caps_nvmm = Gst.ElementFactory.make("capsfilter", f"nvmm_caps_{i}")
        caps_nvmm.set_property('caps', Gst.Caps.from_string("video/x-raw(memory:NVMM)"))
        
        pipeline.add(source)
        pipeline.add(capsfilter)
        pipeline.add(vidconv)
        pipeline.add(nvvidconv)
        pipeline.add(caps_nvmm)
        
        source.link(capsfilter)
        capsfilter.link(vidconv)
        vidconv.link(nvvidconv)
        nvvidconv.link(caps_nvmm)
        
        sinkpad = streammux.get_request_pad(f"sink_{i}")
        srcpad = caps_nvmm.get_static_pad("src")
        srcpad.link(sinkpad)
        
        sources.append(source)
    
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    pgie.set_property('config-file-path', "../../configs/config_primary_yolov11.txt")
    
    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    tiler.set_property('rows', 2)
    tiler.set_property('columns', 2)
    tiler.set_property('width', 1920)
    tiler.set_property('height', 1080)

    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    
    sink = Gst.ElementFactory.make("nvoverlaysink", "nvvideo-renderer") if not is_aarch64() else Gst.ElementFactory.make("nv3dsink", "nv3d-sink")
    sink.set_property('sync', 0)
    
    
    pipeline.add(pgie)
    pipeline.add(tiler)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(sink)
    
    streammux.link(pgie)
    pgie.link(tiler)
    tiler.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(sink)
    
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    osdsinkpad = nvosd.get_static_pad("sink")
    if not osdsinkpad:
        sys.stderr.write(" Unable to get sink pad of nvosd \n")

    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)
    
    print("Starting pipeline\n")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    
    pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    main()
