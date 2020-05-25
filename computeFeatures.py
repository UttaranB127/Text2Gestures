import numpy as np
from utils import angle_between
from utils import distance_between
from utils import area_triangle

# Volume of the bounding box
def computeFeature0PerFrame(frame):
	minx = float('inf')
	miny = float('inf')
	minz = float('inf')

	maxx = float('-inf')
	maxy = float('-inf')
	maxz = float('-inf')
	for i in range(16):
		if minx > frame[3*i]:
			minx = frame[3*i]
		elif maxx < frame[3*i]:
			maxx = frame[3*i]

		if miny > frame[3*i + 1]:
			miny = frame[3*i + 1]
		elif maxy < frame[3*i + 1]:
			maxy = frame[3*i + 1]

		if minz > frame[3*i + 2]:
			minz = frame[3*i + 2]
		elif maxz < frame[3*i + 2]:
			maxz = frame[3*i + 2]
	volume = (maxx - minx)*(maxy - miny)*(maxz - minz)
	volume = volume/1000
	return volume

def computeFeature0(frames):
	array = []
	for frame in frames:
		array.append(computeFeature0PerFrame(frame))
	array = np.asarray(array)
	return np.mean(array)

# Angle at neck by shoulders
def computeFeature1PerFrame(frame):
	jid = 4
	rShoulder = np.asarray([frame[3*jid], frame[3*jid + 1], frame[3*jid + 2]])
	jid = 2
	neck = np.asarray([frame[3*jid], frame[3*jid + 1], frame[3*jid + 2]])
	jid = 7
	lShoulder = np.asarray([frame[3*jid], frame[3*jid + 1], frame[3*jid + 2]])
	angle = angle_between(rShoulder-neck, lShoulder-neck)
	return angle

def computeFeature1(frames):
	array = []
	for frame in frames:
		array.append(computeFeature1PerFrame(frame))
	array = np.asarray(array)
	return np.mean(array)

# Angle at right shoulder by neck and left shoulder
def computeFeature2PerFrame(frame):
	jid = 4
	rShoulder = np.asarray([frame[3*jid], frame[3*jid + 1], frame[3*jid + 2]])
	jid = 2
	neck = np.asarray([frame[3*jid], frame[3*jid + 1], frame[3*jid + 2]])
	jid = 7
	lShoulder = np.asarray([frame[3*jid], frame[3*jid + 1], frame[3*jid + 2]])
	angle = angle_between(neck - rShoulder, lShoulder - rShoulder)
	return angle

def computeFeature2(frames):
	array = []
	for frame in frames:
		array.append(computeFeature2PerFrame(frame))
	array = np.asarray(array)
	return np.mean(array)

# Angle at left shoulder by neck and right shoulder
def computeFeature3PerFrame(frame):
	jid = 4
	rShoulder = np.asarray([frame[3*jid], frame[3*jid + 1], frame[3*jid + 2]])
	jid = 2
	neck = np.asarray([frame[3*jid], frame[3*jid + 1], frame[3*jid + 2]])
	jid = 7
	lShoulder = np.asarray([frame[3*jid], frame[3*jid + 1], frame[3*jid + 2]])
	angle = angle_between(neck - lShoulder, rShoulder - lShoulder)
	return angle

def computeFeature3(frames):
	array = []
	for frame in frames:
		array.append(computeFeature3PerFrame(frame))
	array = np.asarray(array)
	return np.mean(array)

# Angle at neck by vertical and back
def computeFeature4PerFrame(frame):
	jid = 3
	head = np.asarray([frame[3*jid], frame[3*jid + 1], frame[3*jid + 2]])
	jid = 0
	root = np.asarray([frame[3*jid], frame[3*jid + 1], frame[3*jid + 2]])
	up = np.asarray([0.0, 1.0, 0.0])
	angle = angle_between(head - root, up)
	return angle

def computeFeature4(frames):
	array = []
	for frame in frames:
		array.append(computeFeature4PerFrame(frame))
	array = np.asarray(array)
	return np.mean(array)

# Angle at neck by head and back
def computeFeature5PerFrame(frame):
	jid = 3
	head = np.asarray([frame[3*jid], frame[3*jid + 1], frame[3*jid + 2]])
	jid = 2
	neck = np.asarray([frame[3*jid], frame[3*jid + 1], frame[3*jid + 2]])
	jid = 1
	spine = np.asarray([frame[3*jid], frame[3*jid + 1], frame[3*jid + 2]])
	angle = angle_between(head - neck, spine - neck)
	return angle

def computeFeature5(frames):
	array = []
	for frame in frames:
		array.append(computeFeature5PerFrame(frame))
	array = np.asarray(array)
	return np.mean(array)

# Distance between right hand and root
def computeFeature6PerFrame(frame):
	jid = 6
	hand = np.asarray([frame[3*jid], frame[3*jid + 1], frame[3*jid + 2]])
	jid = 0
	root = np.asarray([frame[3*jid], frame[3*jid + 1], frame[3*jid + 2]])
	distance = distance_between(hand, root)
	return distance/10

def computeFeature6(frames):
	array = []
	for frame in frames:
		array.append(computeFeature6PerFrame(frame))
	array = np.asarray(array)
	return np.mean(array)

# Distance between left hand and root
def computeFeature7PerFrame(frame):
	jid = 9
	hand = np.asarray([frame[3*jid], frame[3*jid + 1], frame[3*jid + 2]])
	jid = 0
	root = np.asarray([frame[3*jid], frame[3*jid + 1], frame[3*jid + 2]])
	distance = distance_between(hand, root)
	return distance/10

def computeFeature7(frames):
	array = []
	for frame in frames:
		array.append(computeFeature7PerFrame(frame))
	array = np.asarray(array)
	return np.mean(array)

# Distance between right foot and root
def computeFeature8PerFrame(frame):
	jid = 12
	foot = np.asarray([frame[3*jid], frame[3*jid + 1], frame[3*jid + 2]])
	jid = 0
	root = np.asarray([frame[3*jid], frame[3*jid + 1], frame[3*jid + 2]])
	distance = distance_between(foot, root)
	return distance/10

def computeFeature8(frames):
	array = []
	for frame in frames:
		array.append(computeFeature8PerFrame(frame))
	array = np.asarray(array)
	return np.mean(array)

# Distance between left foot and root
def computeFeature9PerFrame(frame):
	jid = 15
	foot = np.asarray([frame[3*jid], frame[3*jid + 1], frame[3*jid + 2]])
	jid = 0
	root = np.asarray([frame[3*jid], frame[3*jid + 1], frame[3*jid + 2]])
	distance = distance_between(foot, root)
	return distance/10

def computeFeature9(frames):
	array = []
	for frame in frames:
		array.append(computeFeature9PerFrame(frame))
	array = np.asarray(array)
	return np.mean(array)

# Area of triangle between hands and neck
def computeFeature10PerFrame(frame):
	jid = 9
	lHand = np.asarray([frame[3*jid], frame[3*jid + 1], frame[3*jid + 2]])
	jid = 2
	neck = np.asarray([frame[3*jid], frame[3*jid + 1], frame[3*jid + 2]])
	jid = 6
	rHand = np.asarray([frame[3*jid], frame[3*jid + 1], frame[3*jid + 2]])
	area = area_triangle(lHand, neck, rHand)
	return area/100

def computeFeature10(frames):
	array = []
	for frame in frames:
		array.append(computeFeature10PerFrame(frame))
	array = np.asarray(array)
	return np.mean(array)

# Area of triangle between feet and root
def computeFeature11PerFrame(frame):
	jid = 15
	lFoot = np.asarray([frame[3*jid], frame[3*jid + 1], frame[3*jid + 2]])
	jid = 0
	root = np.asarray([frame[3*jid], frame[3*jid + 1], frame[3*jid + 2]])
	jid = 12
	rFoot = np.asarray([frame[3*jid], frame[3*jid + 1], frame[3*jid + 2]])
	area = area_triangle(lFoot, root, rFoot)
	return area/100

def computeFeature11(frames):
	array = []
	for frame in frames:
		array.append(computeFeature11PerFrame(frame))
	array = np.asarray(array)
	return np.mean(array)

# Speed of right hand
def computeFeature12(frames, timestep):
	array = []
	jid = 6
	oldPosition = np.asarray([frames[0][3*jid], frames[0][3*jid + 1], frames[0][3*jid + 2]])
	for i in range(1, len(frames)):
		newPosition = np.asarray([frames[i][3*jid], frames[i][3*jid + 1], frames[i][3*jid + 2]])
		distance = distance_between(newPosition, oldPosition)/10
		array.append(distance/timestep)
		oldPosition = newPosition
	array = np.asarray(array)
	return np.mean(array)

# Speed of left hand
def computeFeature13(frames, timestep):
	array = []
	jid = 9
	oldPosition = np.asarray([frames[0][3*jid], frames[0][3*jid + 1], frames[0][3*jid + 2]])
	for i in range(1, len(frames)):
		newPosition = np.asarray([frames[i][3*jid], frames[i][3*jid + 1], frames[i][3*jid + 2]])
		distance = distance_between(newPosition, oldPosition)/10
		array.append(distance/timestep)
		oldPosition = newPosition
	array = np.asarray(array)
	return np.mean(array)

# Speed of head
def computeFeature14(frames, timestep):
	array = []
	jid = 3
	oldPosition = np.asarray([frames[0][3*jid], frames[0][3*jid + 1], frames[0][3*jid + 2]])
	for i in range(1, len(frames)):
		newPosition = np.asarray([frames[i][3*jid], frames[i][3*jid + 1], frames[i][3*jid + 2]])
		distance = distance_between(newPosition, oldPosition)/10
		array.append(distance/timestep)
		oldPosition = newPosition
	array = np.asarray(array)
	return np.mean(array)

# Speed of right foot
def computeFeature15(frames, timestep):
	array = []
	jid = 12
	oldPosition = np.asarray([frames[0][3*jid], frames[0][3*jid + 1], frames[0][3*jid + 2]])
	for i in range(1, len(frames)):
		newPosition = np.asarray([frames[i][3*jid], frames[i][3*jid + 1], frames[i][3*jid + 2]])
		distance = distance_between(newPosition, oldPosition)/10
		array.append(distance/timestep)
		oldPosition = newPosition
	array = np.asarray(array)
	return np.mean(array)

# Speed of left foot
def computeFeature16(frames, timestep):
	array = []
	jid = 15
	oldPosition = np.asarray([frames[0][3*jid], frames[0][3*jid + 1], frames[0][3*jid + 2]])
	for i in range(1, len(frames)):
		newPosition = np.asarray([frames[i][3*jid], frames[i][3*jid + 1], frames[i][3*jid + 2]])
		distance = distance_between(newPosition, oldPosition)/10
		array.append(distance/timestep)
		oldPosition = newPosition
	array = np.asarray(array)
	return np.mean(array)

# Acceleration of right hand
def computeFeature17(frames, timestep):
	array = []
	jid = 6
	oldPosition = np.asarray([frames[0][3*jid], frames[0][3*jid + 1], frames[0][3*jid + 2]])
	newPosition = np.asarray([frames[1][3*jid], frames[1][3*jid + 1], frames[1][3*jid + 2]])
	oldVelocity = (newPosition - oldPosition)/timestep
	oldPosition = newPosition 
	for i in range(2, len(frames)):
		newPosition = np.asarray([frames[i][3*jid], frames[i][3*jid + 1], frames[i][3*jid + 2]])
		newVelocity = (newPosition - oldPosition)/timestep
		acceleration = (newVelocity - oldVelocity)/timestep
		accelerationMag = np.linalg.norm(acceleration)/10
		oldPosition = newPosition
		oldVelocity = newVelocity
		array.append(accelerationMag)
	array = np.asarray(array)
	return np.mean(array)

# Acceleration of left hand
def computeFeature18(frames, timestep):
	array = []
	jid = 9
	oldPosition = np.asarray([frames[0][3*jid], frames[0][3*jid + 1], frames[0][3*jid + 2]])
	newPosition = np.asarray([frames[1][3*jid], frames[1][3*jid + 1], frames[1][3*jid + 2]])
	oldVelocity = (newPosition - oldPosition)/timestep
	oldPosition = newPosition 
	for i in range(2, len(frames)):
		newPosition = np.asarray([frames[i][3*jid], frames[i][3*jid + 1], frames[i][3*jid + 2]])
		newVelocity = (newPosition - oldPosition)/timestep
		acceleration = (newVelocity - oldVelocity)/timestep
		accelerationMag = np.linalg.norm(acceleration)/10
		oldPosition = newPosition
		oldVelocity = newVelocity
		array.append(accelerationMag)
	array = np.asarray(array)
	return np.mean(array)

# Acceleration of head
def computeFeature19(frames, timestep):
	array = []
	jid = 3
	oldPosition = np.asarray([frames[0][3*jid], frames[0][3*jid + 1], frames[0][3*jid + 2]])
	newPosition = np.asarray([frames[1][3*jid], frames[1][3*jid + 1], frames[1][3*jid + 2]])
	oldVelocity = (newPosition - oldPosition)/timestep
	oldPosition = newPosition 
	for i in range(2, len(frames)):
		newPosition = np.asarray([frames[i][3*jid], frames[i][3*jid + 1], frames[i][3*jid + 2]])
		newVelocity = (newPosition - oldPosition)/timestep
		acceleration = (newVelocity - oldVelocity)/timestep
		accelerationMag = np.linalg.norm(acceleration)/10
		oldPosition = newPosition
		oldVelocity = newVelocity
		array.append(accelerationMag)
	array = np.asarray(array)
	return np.mean(array)

# Acceleration of right foot
def computeFeature20(frames, timestep):
	array = []
	jid = 12
	oldPosition = np.asarray([frames[0][3*jid], frames[0][3*jid + 1], frames[0][3*jid + 2]])
	newPosition = np.asarray([frames[1][3*jid], frames[1][3*jid + 1], frames[1][3*jid + 2]])
	oldVelocity = (newPosition - oldPosition)/timestep
	oldPosition = newPosition 
	for i in range(2, len(frames)):
		newPosition = np.asarray([frames[i][3*jid], frames[i][3*jid + 1], frames[i][3*jid + 2]])
		newVelocity = (newPosition - oldPosition)/timestep
		acceleration = (newVelocity - oldVelocity)/timestep
		accelerationMag = np.linalg.norm(acceleration)/10
		oldPosition = newPosition
		oldVelocity = newVelocity
		array.append(accelerationMag)
	array = np.asarray(array)
	return np.mean(array)

# Acceleration of left foot
def computeFeature21(frames, timestep):
	array = []
	jid = 15
	oldPosition = np.asarray([frames[0][3*jid], frames[0][3*jid + 1], frames[0][3*jid + 2]])
	newPosition = np.asarray([frames[1][3*jid], frames[1][3*jid + 1], frames[1][3*jid + 2]])
	oldVelocity = (newPosition - oldPosition)/timestep
	oldPosition = newPosition 
	for i in range(2, len(frames)):
		newPosition = np.asarray([frames[i][3*jid], frames[i][3*jid + 1], frames[i][3*jid + 2]])
		newVelocity = (newPosition - oldPosition)/timestep
		acceleration = (newVelocity - oldVelocity)/timestep
		accelerationMag = np.linalg.norm(acceleration)/10
		oldPosition = newPosition
		oldVelocity = newVelocity
		array.append(accelerationMag)
	array = np.asarray(array)
	return np.mean(array)

# Movement jerk of right hand
def computeFeature22(frames, timestep):
	array = []
	jid = 6
	oldPosition = np.asarray([frames[0][3*jid], frames[0][3*jid + 1], frames[0][3*jid + 2]])
	newPosition = np.asarray([frames[1][3*jid], frames[1][3*jid + 1], frames[1][3*jid + 2]])
	oldVelocity = (newPosition - oldPosition)/timestep
	oldPosition = newPosition	
	newPosition = np.asarray([frames[2][3*jid], frames[2][3*jid + 1], frames[2][3*jid + 2]])
	newVelocity = (newPosition - oldPosition)/timestep
	oldAcceleration = (newVelocity - oldVelocity)/timestep
	oldVelocity = newVelocity
	oldPosition = newPosition
	for i in range(3, len(frames)):
		newPosition = np.asarray([frames[i][3*jid], frames[i][3*jid + 1], frames[i][3*jid + 2]])
		newVelocity = (newPosition - oldPosition)/timestep
		newAcceleration = (newVelocity - oldVelocity)/timestep
		jerk = (newAcceleration - oldAcceleration)/timestep
		jerkMag = np.linalg.norm(jerk)/10
		oldPosition = newPosition
		oldVelocity = newVelocity
		oldAcceleration = newAcceleration
		array.append(jerkMag)
	array = np.asarray(array)
	return np.mean(array)

# Movement jerk of left hand
def computeFeature23(frames, timestep):
	array = []
	jid = 9
	oldPosition = np.asarray([frames[0][3*jid], frames[0][3*jid + 1], frames[0][3*jid + 2]])
	newPosition = np.asarray([frames[1][3*jid], frames[1][3*jid + 1], frames[1][3*jid + 2]])
	oldVelocity = (newPosition - oldPosition)/timestep
	oldPosition = newPosition	
	newPosition = np.asarray([frames[2][3*jid], frames[2][3*jid + 1], frames[2][3*jid + 2]])
	newVelocity = (newPosition - oldPosition)/timestep
	oldAcceleration = (newVelocity - oldVelocity)/timestep
	oldVelocity = newVelocity
	oldPosition = newPosition
	for i in range(3, len(frames)):
		newPosition = np.asarray([frames[i][3*jid], frames[i][3*jid + 1], frames[i][3*jid + 2]])
		newVelocity = (newPosition - oldPosition)/timestep
		newAcceleration = (newVelocity - oldVelocity)/timestep
		jerk = (newAcceleration - oldAcceleration)/timestep
		jerkMag = np.linalg.norm(jerk)/10
		oldPosition = newPosition
		oldVelocity = newVelocity
		oldAcceleration = newAcceleration
		array.append(jerkMag)
	array = np.asarray(array)
	return np.mean(array)

# Movement jerk of head
def computeFeature24(frames, timestep):
	array = []
	jid = 3
	oldPosition = np.asarray([frames[0][3*jid], frames[0][3*jid + 1], frames[0][3*jid + 2]])
	newPosition = np.asarray([frames[1][3*jid], frames[1][3*jid + 1], frames[1][3*jid + 2]])
	oldVelocity = (newPosition - oldPosition)/timestep
	oldPosition = newPosition	
	newPosition = np.asarray([frames[2][3*jid], frames[2][3*jid + 1], frames[2][3*jid + 2]])
	newVelocity = (newPosition - oldPosition)/timestep
	oldAcceleration = (newVelocity - oldVelocity)/timestep
	oldVelocity = newVelocity
	oldPosition = newPosition
	for i in range(3, len(frames)):
		newPosition = np.asarray([frames[i][3*jid], frames[i][3*jid + 1], frames[i][3*jid + 2]])
		newVelocity = (newPosition - oldPosition)/timestep
		newAcceleration = (newVelocity - oldVelocity)/timestep
		jerk = (newAcceleration - oldAcceleration)/timestep
		jerkMag = np.linalg.norm(jerk)/10
		oldPosition = newPosition
		oldVelocity = newVelocity
		oldAcceleration = newAcceleration
		array.append(jerkMag)
	array = np.asarray(array)
	return np.mean(array)

# Movement jerk of right foot
def computeFeature25(frames, timestep):
	array = []
	jid = 12
	oldPosition = np.asarray([frames[0][3*jid], frames[0][3*jid + 1], frames[0][3*jid + 2]])
	newPosition = np.asarray([frames[1][3*jid], frames[1][3*jid + 1], frames[1][3*jid + 2]])
	oldVelocity = (newPosition - oldPosition)/timestep
	oldPosition = newPosition	
	newPosition = np.asarray([frames[2][3*jid], frames[2][3*jid + 1], frames[2][3*jid + 2]])
	newVelocity = (newPosition - oldPosition)/timestep
	oldAcceleration = (newVelocity - oldVelocity)/timestep
	oldVelocity = newVelocity
	oldPosition = newPosition
	for i in range(3, len(frames)):
		newPosition = np.asarray([frames[i][3*jid], frames[i][3*jid + 1], frames[i][3*jid + 2]])
		newVelocity = (newPosition - oldPosition)/timestep
		newAcceleration = (newVelocity - oldVelocity)/timestep
		jerk = (newAcceleration - oldAcceleration)/timestep
		jerkMag = np.linalg.norm(jerk)/10
		oldPosition = newPosition
		oldVelocity = newVelocity
		oldAcceleration = newAcceleration
		array.append(jerkMag)
	array = np.asarray(array)
	return np.mean(array)

# Movement jerk of left foot
def computeFeature26(frames, timestep):
	array = []
	jid = 15
	oldPosition = np.asarray([frames[0][3*jid], frames[0][3*jid + 1], frames[0][3*jid + 2]])
	newPosition = np.asarray([frames[1][3*jid], frames[1][3*jid + 1], frames[1][3*jid + 2]])
	oldVelocity = (newPosition - oldPosition)/timestep
	oldPosition = newPosition	
	newPosition = np.asarray([frames[2][3*jid], frames[2][3*jid + 1], frames[2][3*jid + 2]])
	newVelocity = (newPosition - oldPosition)/timestep
	oldAcceleration = (newVelocity - oldVelocity)/timestep
	oldVelocity = newVelocity
	oldPosition = newPosition
	for i in range(3, len(frames)):
		newPosition = np.asarray([frames[i][3*jid], frames[i][3*jid + 1], frames[i][3*jid + 2]])
		newVelocity = (newPosition - oldPosition)/timestep
		newAcceleration = (newVelocity - oldVelocity)/timestep
		jerk = (newAcceleration - oldAcceleration)/timestep
		jerkMag = np.linalg.norm(jerk)/10
		oldPosition = newPosition
		oldVelocity = newVelocity
		oldAcceleration = newAcceleration
		array.append(jerkMag)
	array = np.asarray(array)
	return np.mean(array)

def computeFeatures(frames, timestep):
	features = []
	# Volume
	features.append(computeFeature0(frames))
	# Angles
	features.append(computeFeature1(frames))
	features.append(computeFeature2(frames))
	features.append(computeFeature3(frames))
	features.append(computeFeature4(frames))
	features.append(computeFeature5(frames))
	# Distances
	features.append(computeFeature6(frames))
	features.append(computeFeature7(frames))
	features.append(computeFeature8(frames))
	features.append(computeFeature9(frames))
	# Areas
	features.append(computeFeature10(frames))
	features.append(computeFeature11(frames))
	# Speeds
	features.append(computeFeature12(frames, timestep))
	features.append(computeFeature13(frames, timestep))
	features.append(computeFeature14(frames, timestep))
	features.append(computeFeature15(frames, timestep))
	features.append(computeFeature16(frames, timestep))
	# Accelerations
	features.append(computeFeature17(frames, timestep))
	features.append(computeFeature18(frames, timestep))
	features.append(computeFeature19(frames, timestep))
	features.append(computeFeature20(frames, timestep))
	features.append(computeFeature21(frames, timestep))
	# Movement Jerk
	features.append(computeFeature22(frames, timestep))
	features.append(computeFeature23(frames, timestep))
	features.append(computeFeature24(frames, timestep))
	features.append(computeFeature25(frames, timestep))
	features.append(computeFeature26(frames, timestep))
	return features