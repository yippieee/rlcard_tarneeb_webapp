import flask
from flask import request, jsonify
import subprocess
import time			 #You don't need this. Just included it so you can see the output stream.

proc = subprocess.Popen(
	['python', 'demo_v2_gameplay.py'],
	# ['python', 'hello.py'],
	# shell=True,
	stdout=subprocess.PIPE,
	stdin=subprocess.PIPE
)

app = flask.Flask(__name__)

@app.route('/write_stdin', methods=['POST'])
def write_stdin():
	x = request.form.get('choice') + '\n'
	# print('x=', x)
	proc.stdin.write(str.encode(x))
	proc.stdin.flush()
	return flask.Response('-', mimetype='text/html') 
	

@app.route('/play', methods=['GET', 'POST'])
def index():
	def inner():
		#for line in iter(proc.stdout.readline,''):
		# import pdb; pdb.set_trace()
		for line in proc.stdout:
			time.sleep(.01)
			#import pdb; pdb.set_trace()
			output = line.decode()
			yield output + '<br/>\n'
			# print('#', output)
			# print(output, '\t', output.strip()=='>> You choose action (integer):')
			if output.strip()=='>> You choose action (integer):':
				# if output.strip()=='Enter:':
				yield "<html><body><iframe name='dummyframe' id='dummyframe' style='display: none;'></iframe><form action='write_stdin' method='POST' target='dummyframe'><input type='text' name='choice'><input type='submit'></form></body></html>"
				
				#x = input('>> You choose action (integer):\n') + '\n'
				#print('x=', x)
				# proc.stdin.write(str.encode(x))
				# proc.stdin.flush()

		# '''
	return flask.Response(inner(), mimetype='text/html') 
	# return flask.Response(inner(), mimetype='html')

app.run(debug=True, port=5000, host='0.0.0.0')
