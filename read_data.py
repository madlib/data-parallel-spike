import sys
import numpy as np

class ReadData:
	def __init__(self):
		self.f = open('/tmp/read_data.log', 'a')
		self.x_train, self.y_train = [], []
		self.NUMERIC = set(['smallint', 'integer', 'bigint', 'decimal', 'numeric',
			'real', 'double precision', 'serial', 'bigserial'])
		self.INTEGER = set(['smallint', 'integer', 'bigint'])
		self.TEXT = set(['text', 'varchar', 'character varying', 'char', 'character'])
		self.BOOLEAN = set(['boolean'])
		self.read_from_pipe()
		self.f.write("\n")
		self.f.close()

	def read_from_pipe(self):
		for line in sys.stdin:
			self.f.write(line)
			self.f.write("\n")
			split_line = line.split('|')
			x_train_row = split_line[0].strip()
			y_train_row = split_line[1].strip()
			x_train_row_type = split_line[2].strip()
			y_train_row_type = split_line[3].strip()
			x_train_row = self.get_converted_value(x_train_row_type, x_train_row)
			y_train_row = self.get_converted_value(y_train_row_type, y_train_row)
			self.x_train.append(x_train_row)
			self.y_train.append(y_train_row)

		self.x_train = np.array(self.x_train)
		self.y_train = np.array(self.y_train)

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