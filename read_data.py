import sys
import numpy as np
import traceback
import os

if len(sys.argv) != 2:
	raise Exception("module name is missing")
user_module_name = sys.argv[1]
user_function = getattr(__import__(user_module_name), 'some_function_name')

class UserResult:
	def __init__(self, user_result, segment_id, grp_col):
		self.user_result = user_result
		self.segment_id = segment_id
		self.grp_col = grp_col

	def write_to_file(self):
		"""
		what is faster ?
			writing to file on disk and then reading through an external table
		or
			writing to file on disk and then running the `copy from` command
		add segment id to model file name to make it unique.
		we don't need to open the file in write mode since each segment will
		write to it's own file.

		Another approach
			instead of writing to file in python, we can return the UserResult class
			and then cat the output to a file. SO the fit function would look like this

			CREATE WRITABLE EXTERNAL WEB TABLE ext (...)
			EXECUTE 'python /home/gpadmin/read_data.py keras_train > /tmp/user_result_${GP_SEGMENT_ID}' ... ;

		:return:
		"""

		result_file_name = '/tmp/user_result_{0}'.format(self.segment_id)
		result_file = open(result_file_name, 'w')
		if self.grp_col:
			final_result = ','.join([str(self.user_result), self.segment_id, self.grp_col])
		else:
			final_result = ','.join([str(self.user_result), self.segment_id])
		result_file.write(final_result)
		result_file.write("\n")
		result_file.close()

class ReadData:
	"""
	* should all the log file names be unique to each segment?
	* location for log files and the user result file ?
		* madlib folder on master and segment host? This script doesn't know where
			madlib is installed.
		* default gpAdmingLogs or $MASTER_DATA_DIRECTORY ?
	"""
	def __init__(self):
		#TODO error check if GP_SEGMENT_ID is not set
		self.segment_id = os.environ['GP_SEGMENT_ID']
		self.log_file = open('/tmp/read_data_{0}.log'.format(self.segment_id), 'w')
		self.log_file.write("Starting read data \n")

		self.NUMERIC = set(['smallint', 'integer', 'bigint', 'decimal', 'numeric',
			'real', 'double precision', 'serial', 'bigserial'])
		self.INTEGER = set(['smallint', 'integer', 'bigint'])
		self.TEXT = set(['text', 'varchar', 'character varying', 'char', 'character'])
		self.BOOLEAN = set(['boolean'])

		self.run_user_module()

		self.log_file.write("\n")
		self.log_file.close()

	def run_user_module(self,):
		data = self.read_from_pipe()
		run_user_fn_log = open('/tmp/run_user_function_{0}.log'.format(self.segment_id), 'w')
		run_user_fn_log.write("running user module \n")
		try:
			# add error handling
			run_user_fn_result = user_function(data)
			result = UserResult(run_user_fn_result, self.segment_id, None)
			result.write_to_file()
		except:
			run_user_fn_log.write(str(traceback.format_exc()))
			run_user_fn_log.close()
			raise
		run_user_fn_log.close()

	def read_from_pipe(self):
		"""
		* how do other products do type mapping?
		* read from metadata to parse the input line
		* sanity check
			input line should match the metadata
			for ex throw an error if we are expecting integer[] but we get a scalar value
		"""
		try:
			x_train, y_train = [], []
			data = dict()
			for line in sys.stdin:
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
		except:
			self.log_file.write(str(traceback.format_exc()))
			raise
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
			return bool(value)
		elif pg_type in self.TEXT:
			return value

		raise Exception("unsupported scalar type {0}".format(pg_type))

	def get_converted_array_value(self, pg_type, value):
		# TODO should we assume that all arrays are numeric in the form {1,2...}
		return np.array(map(float, value[1:len(value)-1].split(',')))

	#TODO implementation of this func is TBD. Writing metadata to a file is one way.
	# explore other avenues.
	def get_metadata(self):
		"""
		* We need to know the following metadata
			* column types, names and indexes in the tuple
			* grouping col name and index in the tuple
		"""
		# error handling if metadata file does not exist
		metadata_file = open('/tmp/metadata', 'r')
		# read from yaml
		metadata_file.close()
		return None

ReadData()

# how to test read_data.py
#echo '{1,2,3}|2|integer[]|integer' | GP_SEGMENT_ID=2 python read_data.py keras_train
