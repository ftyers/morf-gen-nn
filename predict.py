"""
Learn a model to morphologically generate surface forms from analyses

In this demo, a recurrent network equipped with an attention mechanism
learns to morphologically generate each word (on a symbol-by-symbol basis) in
its input text.

Mostly hacked from WordReverser

"""

import sys, math, numpy, operator;

from theano import tensor

from blocks.bricks import Tanh, Initializable
from blocks.bricks.base import application
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import SimpleRecurrent, Bidirectional
from blocks.bricks.attention import SequenceContentAttention
from blocks.bricks.parallel import Fork
from blocks.bricks.sequence_generators import (SequenceGenerator, Readout, SoftmaxEmitter, LookupFeedback)

from fuel.datasets import TextFile
from fuel.transformers import Mapping, Batch, Padding, Filter
from fuel.schemes import ConstantScheme

from blocks.initialization import Orthogonal, IsotropicGaussian, Constant
from blocks.utils import dict_union
from blocks.model import Model
from blocks.graph import ComputationGraph
from blocks.algorithms import (GradientDescent, Scale, StepClipping, CompositeRule)
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Printing, Timing
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import TrainingDataMonitoring
from blocks.main_loop import MainLoop

##

from blocks.serialization import load_parameters
from blocks.filter import VariableFilter
from picklable_itertools.extras import equizip
from blocks.search import BeamSearch





class Globals: #{

	char2code = {};
	code2char = {};
	lookup = {};

	def read_alphabet(path): #{
		f = open(path);
		vocab = f.readline().strip();
		for line in f.readlines(): #{
			row = line.strip().split('\t');
			code = int(row[0]);
			sym = row[1];
			Globals.char2code[sym] = code; 
			Globals.code2char[code] = sym; 
		#}
		f.close();
	#}

	def read_lookup(path): #{

		f = open(path);
		for line in f.readlines(): #{
			row = line.strip().replace('|||', '\t').split('\t');

			out_string = '<S>';
			out_symbols = []; out_symbols.append(Globals.char2code['<S>']);
			for c in row[0]: #{
				out_string = out_string + ' ' + c ;
				out_symbols.append(Globals.char2code[c]);
			#}	
			out_string = out_string + ' ' + '</S>';
			out_symbols.append(Globals.char2code['</S>']);

			in_string = '<S>';
			in_symbols = []; in_symbols.append(Globals.char2code['<S>']);
			for c in row[1]: #{
				in_string = in_string + ' ' + c ;
				in_symbols.append(Globals.char2code[c]);
			#}
			for tag in row[2].split('|'): #{
				in_string = in_string + ' ' + tag ;
				in_symbols.append(Globals.char2code[tag]);
			#}
			in_string = in_string + ' ' + '</S>';
			in_symbols.append(Globals.char2code['</S>']);

#			print(in_string,'→', in_symbols, file=sys.stderr);
#			print(out_string,'→', out_symbols, file=sys.stderr);

			Globals.lookup[tuple(in_symbols)] = out_symbols;
		#}
	#}

#}

class MorphGen(Initializable): #{

	def __init__(self, dimen, vocab_size): #{
		# No idea what this is doing, but otherwise "allocated" is not set
		super(MorphGen, self).__init__(self)

		# The encoder 
		encoder = Bidirectional(SimpleRecurrent(dim=dimen, activation=Tanh()))

		# What is this doing ? 
		fork = Fork([name for name in encoder.prototype.apply.sequences if name != 'mask'])
		fork.input_dim = dimen
		fork.output_dims = [encoder.prototype.get_dim(name) for name in fork.input_names]

		lookup = LookupTable(vocab_size, dimen)
		transition = SimpleRecurrent(activation=Tanh(),dim=dimen, name="transition")
		attention = SequenceContentAttention(state_names=transition.apply.states,attended_dim=2*dimen, match_dim=dimen, name="attention")
		readout = Readout(
			readout_dim=vocab_size,
			source_names=[transition.apply.states[0],
			attention.take_glimpses.outputs[0]],
			emitter=SoftmaxEmitter(name="emitter"),
			feedback_brick=LookupFeedback(vocab_size, dimen),
			name="readout");
		generator = SequenceGenerator(readout=readout, transition=transition, attention=attention,name="generator")
	
		self.lookup = lookup
		self.fork = fork
		self.encoder = encoder
		self.generator = generator
		self.children = [lookup, fork, encoder, generator]
	#}

	@application
	def cost(self, chars, chars_mask, targets, targets_mask): #{
		return self.generator.cost_matrix(targets, targets_mask,
			attended=self.encoder.apply(**dict_union(
				self.fork.apply(self.lookup.apply(chars), as_dict=True),mask=chars_mask)),
			attended_mask=chars_mask);
	#}

	@application
	def generate(self, chars): #{
		return self.generator.generate(n_steps=3 * chars.shape[0], batch_size=chars.shape[1],
			attended=self.encoder.apply(**dict_union(
				self.fork.apply(self.lookup.apply(chars), as_dict=True))),
			attended_mask=tensor.ones(chars.shape))
	#}

#}

# What does this do ?
def _transpose(data): #{
    return tuple(array.T for array in data)
#


def _tokenise(s): #{
#	print('');
#	print('@_tokenise()', s.strip(), file=sys.stderr);
	row = s.strip().replace('|||', '\t').split('\t');

	in_string = '';
	for c in row[1]: #{
		in_string = in_string + ' ' + c ;
	#}
	for tag in row[2].split('|'): #{
		in_string = in_string + ' ' + tag ;
	#}

	return in_string;
#}

def _encode(s): #{
#	print('@_encode():', s);
	enc = [];
	enc.append(Globals.char2code['<S>']);
	for c in s.strip().split(' '): #{
		enc.append(Globals.char2code[c]);
	#}	
	enc.append(Globals.char2code['</S>']);

	return enc;
#}

def _decode(l): #{
	dec = [];
	for c in l: #{
		dec.append(Globals.code2char[c]);
	#}	
	return dec;
#}


def morph_lookup(l): #{
#	print('@_morph_lookup()', l[0], file=sys.stderr);
	lkp = tuple(l[0]);
	if lkp in Globals.lookup: #{
#		print('@_morph_lookup()', Globals.lookup[lkp], file=sys.stderr);
		return (Globals.lookup[lkp],);
	else: #{
		for x in Globals.lookup: #{
			print(x,'→', Globals.lookup[x], file=sys.stderr);
		#}
		sys.exit(-1);
	#}
#}

def _is_nan(log): #{
    return math.isnan(log.current_row['total_gradient_norm'])
#}

def generate(m, input_): #{
	samples, = VariableFilter(applications=[m.generator.generate], name="outputs")(ComputationGraph(generated[1]))
	# NOTE: this will recompile beam search functions every time user presses Enter. Do not create
	# a new `BeamSearch` object every time if speed is important for you.
	beam_search = BeamSearch(samples);
	outputs, costs = beam_search.search({chars: input_}, Globals.char2code['</S>'], 3 * input_.shape[0]);
	return outputs, costs;
#}

f_vocab = '';
f_train = '';
f_model = '';
n_batches = '';

BEAM = 10;

if len(sys.argv) < 5: #{
	print('predict.py <vocab> <test> <nbest> <model>');
	sys.exit(-1);
else: #{
	f_vocab = sys.argv[1];
	f_test = sys.argv[2];
	n_best = int(sys.argv[3]);
	f_model = sys.argv[4];
#}

Globals.read_alphabet(f_vocab);
Globals.read_lookup(f_test);

#print("Vocab:",Globals.char2code, file=sys.stderr);
#print("Test:",f_test,file=sys.stderr);

m = MorphGen(100, len(Globals.char2code));

chars = tensor.lmatrix("input")
generated = m.generate(chars)
model = Model(generated)
# Load model
with open(f_model, 'rb') as f: #{
	model.set_parameter_values(load_parameters(f))
#}

f_in = open(f_test);
total = 0.0;
correct = 0.0;
for line in f_in.readlines(): #{
	inp = _tokenise(line);
	encoded_input = _encode(inp);
#	print(inp,'→',encoded_input, sys.stderr);	
	target = morph_lookup((encoded_input,))[0]	
#	print('Target:','→',target, sys.stderr);	

	input_arr = numpy.repeat(numpy.array(encoded_input)[:, None],BEAM, axis=1);
	samples, costs = generate(m, input_arr);

	messages = []
	for sample, cost in equizip(samples, costs): #{
#		message = "({})".format(cost)
		message = "".join(Globals.code2char[code] for code in sample)
		if sample == target: #{
			message += " CORRECT!"
		#}
		messages.append([float(cost), message])
		#messages.sort(key=operator.itemgetter(0), reverse=True)
	#}
	messages.sort()
	for message in messages[0:n_best]: #{
		if 'CORRECT' in message[1]: #{
			correct = correct + 1.0;
		#}
		print(correct/total, message[0], message[1], file=sys.stderr)
	#}

#}
print('%.2f\t%.2f\t%.2f' % (correct/total, total, correct))
