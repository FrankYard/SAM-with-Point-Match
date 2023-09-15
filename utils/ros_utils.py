import numpy as np
import sys
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Image

dtype_mapper = {
	"rgb8":    (np.uint8,  3),
	"rgba8":   (np.uint8,  4),
	"rgb16":   (np.uint16, 3),
	"rgba16":  (np.uint16, 4),
	"bgr8":    (np.uint8,  3),
	"bgra8":   (np.uint8,  4),
	"bgr16":   (np.uint16, 3),
	"bgra16":  (np.uint16, 4),
	"mono8":   (np.uint8,  1),
	"mono16":  (np.uint16, 1),
 
 	"8UC1":    (np.uint8,   1),
	"8UC2":    (np.uint8,   2),
	"8UC3":    (np.uint8,   3),
	"8UC4":    (np.uint8,   4),
	"8SC1":    (np.int8,    1),
	"8SC2":    (np.int8,    2),
	"8SC3":    (np.int8,    3),
	"8SC4":    (np.int8,    4),
	"16UC1":   (np.uint16,   1),
	"16UC2":   (np.uint16,   2),
	"16UC3":   (np.uint16,   3),
	"16UC4":   (np.uint16,   4),
	"16SC1":   (np.int16,  1),
	"16SC2":   (np.int16,  2),
	"16SC3":   (np.int16,  3),
	"16SC4":   (np.int16,  4),
	"32SC1":   (np.int32,   1),
	"32SC2":   (np.int32,   2),
	"32SC3":   (np.int32,   3),
	"32SC4":   (np.int32,   4),
	"32FC1":   (np.float32, 1),
	"32FC2":   (np.float32, 2),
	"32FC3":   (np.float32, 3),
	"32FC4":   (np.float32, 4),
	"64FC1":   (np.float64, 1),
	"64FC2":   (np.float64, 2),
	"64FC3":   (np.float64, 3),
	"64FC4":   (np.float64, 4)
 }


def imgmsg_to_numpy(msg):
	if not msg.encoding in dtype_mapper:
		raise TypeError('Unrecognized encoding {}'.format(msg.encoding))
	
	dtype_class, channels = dtype_mapper[msg.encoding]
	dtype = np.dtype(dtype_class)
	dtype = dtype.newbyteorder('>' if msg.is_bigendian else '<')
	shape = (msg.height, msg.width, channels)

	data = np.fromstring(msg.data, dtype=dtype).reshape(shape)
	data.strides = (
		msg.step,
		dtype.itemsize * channels,
		dtype.itemsize
	)

	return data

def numpy_to_image(arr, encoding):
	if not encoding in dtype_mapper:
		raise TypeError('Unrecognized encoding {}'.format(encoding))

	im = Image(encoding=encoding)

	# extract width, height, and channels
	dtype_class, exp_channels = dtype_mapper[encoding]
	dtype = np.dtype(dtype_class)
	if len(arr.shape) == 2:
		im.height, im.width, channels = arr.shape + (1,)
	elif len(arr.shape) == 3:
		im.height, im.width, channels = arr.shape
	else:
		raise TypeError("Array must be two or three dimensional")

	# check type and channels
	if exp_channels != channels:
		raise TypeError("Array has {} channels, {} requires {}".format(
			channels, encoding, exp_channels
		))
	if dtype_class != arr.dtype.type:
		raise TypeError("Array is {}, {} requires {}".format(
			arr.dtype.type, encoding, dtype_class
		))

	# make the array contiguous in memory, as mostly required by the format
	contig = np.ascontiguousarray(arr)
	im.data = contig.tostring()
	im.step = contig.strides[0]
	im.is_bigendian = (
		arr.dtype.byteorder == '>' or 
		arr.dtype.byteorder == '=' and sys.byteorder == 'big'
	)

	return im

def build_ellipse(frame_id, stamp, parameters):
    """

    """
    msg = MarkerArray()
    for i, (center, scale, quat) in enumerate(parameters):
        marker = Marker()
        marker.header.frame_id = frame_id 
        marker.header.stamp = stamp
        marker.ns = 'object_mark'
        marker.id = i
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.color.a = 0.5
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
    
        marker.pose.position.x = center[0]
        marker.pose.position.y = center[1]
        marker.pose.position.z = center[2]

        marker.pose.orientation.x = quat[0]
        marker.pose.orientation.y = quat[1]
        marker.pose.orientation.z = quat[2]
        marker.pose.orientation.w = quat[3]

        marker.scale.x = scale[0]
        marker.scale.y = scale[1]
        marker.scale.z = scale[2]

        msg.markers.append(marker)
    return msg