from body_measurements.measurement import Body3D
import trimesh
import json
import os
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument('--gender', type = str, required = True,help='male or female')
	parser.add_argument('--path', type = str, required = True, help='path to obj files')
	args = parser.parse_args()
	i=0
						
	gender_measuresDensity = dict()
	gender_measuresSurface = dict()
	gender_measuresVolume = dict()
	measuresDensity = dict()
	measuresSurface = dict()
	measuresVolume = dict()
	exceptionfiles = dict()
	weight=0
	weight2=0
	weight3=0

	for subject in sorted(os.listdir(args.path)):
		filename = os.path.join(args.path, subject)
		print('filename:', filename)
		mesh = trimesh.load(filename)
		try:
			body = Body3D(mesh.vertices, mesh.faces)
			height = body.height()
			print(height)

			weight = body.weight() #weight density
			print(weight)

			# surface = body.surface()
			# weight2 = (36*surface*surface)/height #weight surface
			# print(weight2)

			# weight3 = body.weightVolume() #weight volume
			# print(weight3)


		except:
			print('exception')
			exceptionfiles[i]=filename
			i=i+1
			height = None
			weight = None

		measuresDensity[subject] = [height, weight]
		# measuresSurface[subject] = [height, weight2]
		# measuresVolume[subject] = [height, weight3]
	gender_measuresDensity[args.gender] = measuresDensity
	# gender_measuresSurface[args.gender] = measuresSurface
	# gender_measuresVolume[args.gender] = measuresVolume

	with open(f'h_w_measures_{args.gender}_density_256.json', 'w') as fp:
		print('entrato nella stampa')
		json.dump(gender_measuresDensity, fp)
		print('stampato')
	
	
	# with open(f'h_w_measures_{args.gender}_surface.json', 'w') as fp:
	# 	print('entrato nella stampa')
	# 	json.dump(gender_measuresSurface, fp)
	# 	print('stampato')
	
	# with open(f'h_w_measures_{args.gender}_volume.json', 'w') as fp:
	# 	print('entrato nella stampa')
	# 	json.dump(gender_measuresVolume, fp)
	# 	print('stampato')
	
	# with open(f'h_w_measures_{args.gender}_log.json', 'w') as fp:
	# 	json.dump(exceptionfiles, fp)
	


		
