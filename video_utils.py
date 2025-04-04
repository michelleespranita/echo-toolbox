from decord import VideoReader, cpu

def loadvideo_decord(filepath):
    vr = VideoReader(filepath, num_threads=1, ctx=cpu(0))
    vr.seek(0)
    return vr.get_batch(list(range(len(vr)))).asnumpy()