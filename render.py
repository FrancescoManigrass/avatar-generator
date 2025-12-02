import numpy as np
import trimesh
import pyrender
import matplotlib.pyplot as plt
import os
import argparse
import imageio
import time

def args():

	parser = argparse.ArgumentParser()

	parser.add_argument('--path', type = str, required = True, help='path to the .obj files') 

	parser.add_argument('--gender', type = str, required = True, help='male or female')

	arguments = parser.parse_args()
	return arguments


def render(resolution, file_path, save_path):
	start_time = time.time()

	r = pyrender.OffscreenRenderer(512, 512, point_size=1.0)

	# mesh_front_pose = np.array([
	# 	[1.0, 0.0,  0.0, 0.0],
	# 	[0.0, 1.0,  0.0, 0.5],
	# 	[0.0, 0.0,  1.0, -1.8],
	# 	[0.0, 0.0,  0.0, 1.0],
	# ])

	mesh_front_pose = np.array([
		[1.0, 0.0,  0.0, 0.0],
		[0.0, 1.0,  0.0, 0.48], #valori quarta colonna : #male 0.48     #female 0.48    #0.25 calvis female    #0.25 calvis male
		[0.0, 0.0,  1.0,-6.1],  #valori quarta colonna : #male -6.1     #female -6.1    #-5.8 calvis female    #-6.1 calvis male
		[0.0, 0.0,  0.0, 1.0],
	])

	# mesh_side_rot = np.array([
	# 	[0.0, 0.0, -1.0,  0.0],
	# 	[0.0, 1.0,  0.0,  0.5],
	# 	[1.0, 0.0,  0.0, -1.8],
	# 	[0.0, 0.0,  0.0,  1.0],
	# ])

	# mesh_side_transl = np.array([
	# 	[1.0, 0.0,  0.0,  0.0],
	# 	[0.0, 1.0,  0.0, -0.5],
	# 	[0.0, 0.0,  1.0,  1.8],
	# 	[0.0, 0.0,  0.0,  1.0],
	# ])

	# mesh_side_pose = np.matmul(mesh_side_rot,mesh_side_transl)

	# ortho camera
	# mesh_side_pose = np.array([
	# 	[0.0, 0.0, -1.0, -1.8],
	# 	[0.0, 1.0,  0.0,  0.0],
	# 	[1.0, 0.0,  0.0, -1.8],
	# 	[0.0, 0.0,  0.0,  1.0],
	# ])

	# perspective camera
	mesh_side_pose = np.array([
		[0.0, 0.0, -1.0, -6.1],  #mettere lo stesso valore inserito 4a colonna 3a riga della matrice mesh_front_pose
		[0.0, 1.0,  0.0,  0.0],
		[1.0, 0.0,  0.0, -6.1],  #mettere lo stesso valore inserito 4a colonna 3a riga della matrice mesh_front_pose
		[0.0, 0.0,  0.0,  1.0],
	])

	# s = np.sqrt(2)/2 #0.7
	# camera_pose = np.array([
	#    [0.0,  -s,  s,   0.5],
	#    [1.0,  0.0, 0.4, 0.0],
	#    [0.0,  s,   s,   0.4], 
	#    [0.0,  0.0, 0.0, 1.0],
	# ])

	for subject in os.listdir(file_path):
		_file = os.path.join(file_path, subject)

		_id = subject.split(".")[0].split("_")[-1]

		try:
			os.mkdir(os.path.join(save_path, str(_id)))
		except:
			pass

		try:
			os.mkdir(os.path.join(save_path, _id, str(resolution)))
		except:
			pass

		_folder = os.path.join(save_path, _id, str(resolution))

		fuze_trimesh = trimesh.load(_file)
		
		fuze_trimesh.visual.vertex_colors =[205, 205, 205, -50]
		
		mesh_front = pyrender.Mesh.from_trimesh(fuze_trimesh, poses=mesh_front_pose)

		#scene creation
		scene = pyrender.Scene(ambient_light=np.array([0.02, 0.02, 0.02, 1.0]))
		
		#adding obj to the scene
		msh = pyrender.Node(mesh=mesh_front)
		scene.add_node(msh)
		

		#camera creation

		# Ortho camera
		#camera = pyrender.OrthographicCamera(xmag=0.9, ymag=0.9)

		# Definizione dei parametri della camera
		near = 0.1  # Distanza minima della camera dal primo piano visibile
		far = 100.0  # Distanza massima della camera dal piano più lontano visibile
		# Creazione della camera prospettica (fx,fy - focal lenght) (cx,cy - optical center)
		# Aumentando la focal lenght e avvicinando la mesh diventa sempre più simile alla ortho camera
		camera = pyrender.IntrinsicsCamera(fx=1350, fy=1350, cx=250, cy=250, znear=near, zfar=far)
		
		#adding camera to the scene
		scene.add(camera)
			
		# file name to save
		file_name1 = os.path.join(_folder, 'front.png')
		file_name2 = os.path.join(_folder, 'side.png')

		color, depth = r.render(scene)

		imageio.imwrite(file_name1, color)

		scene.set_pose(msh, pose=mesh_side_pose)
		color, depth = r.render(scene)

		imageio.imwrite(file_name2, color)

		print(subject)
		elapsed_time = time.time() - start_time 
		print(elapsed_time)
		

				
		
def main():
	arguments = args()
	
	gender = arguments.gender

	file_path = arguments.path
	
	resolution = 512

	try:
		os.mkdir("data256")
	except:
		pass
	

	try:
		os.mkdir(f"data256/{gender}")
	except:
		pass
	
	save_path = f"data256/{gender}"
	render(resolution, file_path, save_path)
	

if __name__ == '__main__':
	main()