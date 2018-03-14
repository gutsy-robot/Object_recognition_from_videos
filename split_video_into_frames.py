import cv2
import os, os.path
import cv2

videos_path = os.path.join(os.path.dirname(__file__), 'VIDEOS_720P')

def split_video_into_frames():
	try:
		if not os.path.exists('images'):
			os.makedirs('images')
	except OSError:
		print ('Error: Creating directory of images')
	currentFrame = 0
	for folder in os.listdir(videos_path):
		folder_path = os.path.join(os.path.dirname(__file__), 'VIDEOS_720P/'+folder)
		for video in os.listdir(folder_path):
			print(folder_path)
			print(video)
			cap = cv2.VideoCapture(folder_path+'/'+video)			
			for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
				ret, frame = cap.read()
				name = './images/'+folder+'/frame' + str(i) + '.jpg'
				print ('Creating...' + name)
				cv2.imwrite(name, frame)
				currentFrame += 1
			cap.release()
			cv2.destroyAllWindows()	
		
if __name__ == '__main__':
	split_video_into_frames()