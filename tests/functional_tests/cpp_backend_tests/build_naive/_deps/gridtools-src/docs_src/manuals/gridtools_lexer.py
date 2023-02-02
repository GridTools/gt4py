from pygments.lexer import RegexLexer, inherit, words
from pygments.token import *
from pygments.lexers.c_cpp import CppLexer

gridtools_keywords = ((
    'accessor',
    'in_accessor',
    'inout_accessor',
    'param_list',
    'make_param_list',
    'axis',
    'cells',
    'dimension',
    'edges',
    'extent',
    'fill',
    'flush',
    'global_parameter',
    'intent',
    'layout_map',
    'vertices',
    'halo_descriptor',
    'call',
    'call_proc',
    'with',
    'at',
    'expandable',
))

gridtools_namespace = ((
	'cache_io_policy',
	'cache_type',
    'cartesian',
    'icosahedral',
    'storage',
))

gridtools_functions = ((
    'make_grid',
    'execute_backward',
    'execute_forward',
    'execute_parallel',
    'multi_pass',
    'run',
    'run_single_stage',
    'expandable_run',
	'global_parameter',
    'boundary',
    'halo_exchange_dynamic_ut',
    'halo_exchange_generic',
))

gridtools_macros = ((
	'GT_FUNCTION',
    'GT_DECLARE_TMP',
    'GT_DECLARE_EXPANDABLE_TMP',
))

class GridToolsLexer(CppLexer):
	name = "gridtools"
	aliases = ['gridtools']

	tokens = {
		'statement': [
			(words(gridtools_keywords, suffix=r'\b'), Keyword),
			(words(gridtools_functions, suffix=r'\b'), Name.Label),
			(words(gridtools_namespace, suffix=r'\b'), Name.Namespace),
			(words(gridtools_macros, suffix=r'\b'), Comment.Preproc),
			inherit,
		]
	}
