import sys
import numpy as np

if len(sys.argv) != 2:
	raise Exception("module name is missing")
user_module_name = sys.argv[1]
user_function = getattr(__import__(user_module_name), 'some_function_name')

class ReadData:
	def __init__(self):
		self.seg_id = os.environ['GP_SEGMENT_ID']
		self.log = open('/tmp/read_data.log', 'a')
		self.NUMERIC = set(['smallint', 'integer', 'bigint', 'decimal', 'numeric',
			'real', 'double precision', 'serial', 'bigserial'])
		self.INTEGER = set(['smallint', 'integer', 'bigint'])
		self.TEXT = set(['text', 'varchar', 'character varying', 'char', 'character'])
		self.BOOLEAN = set(['boolean'])
		# self.metadata = self.get_metadata()
		data = self.read_from_pipe()
		user_result = self.run_user_module(data)
		self.write_user_result(result)
		self.log.write("\n")
		self.log.close()

	#implementation of this func is TBD. Writing metadata to a file is one way.
	# explore other avenues.
	def get_metadata(self):
		# error handling if metadata file does not exist
		metadata_file = open('/tmp/metadata', 'r')
		# read from yaml
		metadata_file.close()
		return None

	def run_user_module(self, data):
		try:
			# add error handling
			return user_function(data)
		except:
			f = open('/tmp/user_function.log', 'a')
			f.write(str(traceback.format_exc()))
			f.close()

	def write_user_result(self, user_result):
		# add segment id to result file
		model_file = open('/tmp/local_user_result', 'a')
		model_file.write(str(self.user_result))
		model_file.close()

	def read_from_pipe(self):
		x_train, y_train = [], []
		data = dict()
		for line in sys.stdin:
			self.log.write(line)
			self.log.write("\n")
			split_line = line.split('|')
			x_train_row = split_line[0].strip()
			y_train_row = split_line[1].strip()
			x_train_row_type = split_line[2].strip()
			y_train_row_type = split_line[3].strip()
			x_train_row = self.get_converted_value(x_train_row_type, x_train_row)
			y_train_row = self.get_converted_value(y_train_row_type, y_train_row)
			x_train.append(x_train_row)
			y_train.append(y_train_row)
		x_train = np.array(x_train)
		y_train = np.array(y_train)
		data['x_train'] = x_train
		data['y_train'] = y_train
		return data

	def is_array_type(self, pg_type):
		return pg_type.rstrip().endswith('[]')

	def get_converted_value(self, pg_type, value):
		if self.is_array_type(pg_type):
			return self.get_converted_array_value(pg_type, value)
		else:
			return self.get_converted_scalar_value(pg_type, value)

	def get_converted_scalar_value(self, pg_type, value):
		if pg_type in self.NUMERIC:
			return int(value)
		elif pg_type in self.BOOLEAN:
			return boolean(value)

	def get_converted_array_value(self, pg_type, value):
		# we assume that all arrays are numeric
		return np.array(map(float, value[1:len(value)-1].split(',')))

ReadData()
