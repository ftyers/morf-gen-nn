"""
Learn to morphologically generate surface forms from analyses

In this demo, a recurrent network equipped with an attention mechanism
learns to morphologically generate each word (on a symbol-by-symbol basis) in
its input text.
"""

import sys;

from blocks.bricks import Tanh, Initializable
from blocks.bricks.base import application
from blocks.bricks.lookup import LookupTable
from blocks.bricks.recurrent import SimpleRecurrent, Bidirectional
from blocks.bricks.attention import SequenceContentAttention
from blocks.bricks.parallel import Fork
from blocks.bricks.sequence_generators import (SequenceGenerator, Readout, SoftmaxEmitter, LookupFeedback)



class Globals: #{

	char2code = {};
	code2char = {};
	lookup = {};

	def read_alphabet(path): #{
		f = open(path);
		vocab = f.readline().strip();
		all_chars = (vocab.split(' ') + ['<UNK>']);
		Globals.code2char = dict(enumerate(all_chars))
		Globals.char2code = {v: k for k, v in Globals.code2char.items()}
	#}
#}

class MorphGen: #{

	def __init__(self, dimen, vocab_size): #{
		
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
	            name="readout")
	        generator = SequenceGenerator(readout=readout, transition=transition, attention=attention,name="generator")
	
	        self.lookup = lookup
	        self.fork = fork
	        self.encoder = encoder
	        self.generator = generator
	        self.children = [lookup, fork, encoder, generator]
	
	
	#}
#}

if len(sys.argv) < 2: #{
	sys.exit(-1);
#}

Globals.read_alphabet(sys.argv[1]);

print(Globals.char2code);
print(Globals.code2char);
        
m = MorphGen(100, len(Globals.char2code));


