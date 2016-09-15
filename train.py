"""
Learn a model to morphologically generate surface forms from analyses

In this demo, a recurrent network equipped with an attention mechanism
learns to morphologically generate each word (on a symbol-by-symbol basis) in
its input text.

Mostly hacked from WordReverser

"""

import sys, math;

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

			print(in_string,'→', in_symbols, file=sys.stderr);
			print(out_string,'→', out_symbols, file=sys.stderr);

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

#}

# What does this do ?
def _transpose(data): #{
    return tuple(array.T for array in data)
#


def _tokenise(s): #{
	print('@_tokenise()', s.strip(), file=sys.stderr);
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

f_vocab = '';
f_train = '';
f_model = '';
n_batches = '';

if len(sys.argv) < 5: #{
	print('train.py <vocab> <training> <nbatches> <model>');
	sys.exit(-1);
else: #{
	f_vocab = sys.argv[1];
	f_train = sys.argv[2];
	n_batches = int(sys.argv[3]);
	f_model = sys.argv[4];
#}

Globals.read_alphabet(f_vocab);
Globals.read_lookup(f_train);

print("Vocab:",Globals.char2code, file=sys.stderr);
print("Train:",f_train,file=sys.stderr);

m = MorphGen(100, len(Globals.char2code));

dataset_options = dict(dictionary=Globals.char2code, level="word", preprocess=_tokenise);

dataset = TextFile([f_train], **dataset_options)

data_stream = dataset.get_example_stream()

# Read examples and look up the right surface form
data_stream = Mapping(data_stream, morph_lookup, add_sources=("targets",))

# Read in 10 samples at a time
data_stream = Batch(data_stream, iteration_scheme=ConstantScheme(10)) 

# Pad the examples
data_stream = Padding(data_stream)
data_stream = Mapping(data_stream, _transpose)

# Initialisation settings

m.weights_init = IsotropicGaussian(0.1)
m.biases_init = Constant(0.0)
m.push_initialization_config()
m.encoder.weights_init = Orthogonal()
m.generator.transition.weights_init = Orthogonal()

# Build the cost computation graph
chars = tensor.lmatrix("features")
chars_mask = tensor.matrix("features_mask")
targets = tensor.lmatrix("targets")
targets_mask = tensor.matrix("targets_mask")
batch_cost = m.cost(chars, chars_mask, targets, targets_mask).sum()
batch_size = chars.shape[1].copy(name="batch_size")
cost = aggregation.mean(batch_cost, batch_size)
cost.name = "sequence_log_likelihood"

print("Cost graph is built", file=sys.stderr)

model = Model(cost)
parameters = model.get_parameter_dict()

for brick in model.get_top_bricks(): #{
	brick.initialize();
#}

cg = ComputationGraph(cost);

algo = GradientDescent(cost=cost, parameters=cg.parameters,step_rule=CompositeRule([StepClipping(10.0), Scale(0.01)]))

max_length = chars.shape[0].copy(name="max_length")
observables = [batch_size, max_length,algo.total_step_norm, algo.total_gradient_norm, cost]

# Construct the main loop and start training!
average_monitoring = TrainingDataMonitoring(observables, prefix="average", every_n_batches=10)

main_loop = MainLoop(
	model=model, data_stream=data_stream, algorithm=algo,
            extensions=[
		Timing(),
		TrainingDataMonitoring(observables, after_batch=True),
		average_monitoring,
		FinishAfter(after_n_batches=n_batches)
		# This shows a way to handle NaN emerging during training: simply finish it.
		.add_condition(["after_batch"], _is_nan),
		# Saving the model and the log separately is convenient,
		# because loading the whole pickle takes quite some time.
		Checkpoint(f_model, every_n_batches=5,save_separately=["model", "log"]),
		Printing(every_n_batches=1)]);

main_loop.run()

