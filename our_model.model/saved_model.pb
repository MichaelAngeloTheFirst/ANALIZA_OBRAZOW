ä.
Ö
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.10.12v2.10.0-76-gfdfc646704c8­ð+
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:*
dtype0

Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
à*$
shared_nameAdam/dense/kernel/v
}
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v* 
_output_shapes
:
à*
dtype0

Adam/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*%
shared_nameAdam/conv2d_2/bias/v
y
(Adam/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/v*
_output_shapes
:`*
dtype0

Adam/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0`*'
shared_nameAdam/conv2d_2/kernel/v

*Adam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/v*&
_output_shapes
:0`*
dtype0

Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*%
shared_nameAdam/conv2d_1/bias/v
y
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
_output_shapes
:0*
dtype0

Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*'
shared_nameAdam/conv2d_1/kernel/v

*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*&
_output_shapes
:0*
dtype0
|
Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv2d/bias/v
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes
:*
dtype0

Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/kernel/v

(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
:*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:*
dtype0

Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
à*$
shared_nameAdam/dense/kernel/m
}
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m* 
_output_shapes
:
à*
dtype0

Adam/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*%
shared_nameAdam/conv2d_2/bias/m
y
(Adam/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/m*
_output_shapes
:`*
dtype0

Adam/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0`*'
shared_nameAdam/conv2d_2/kernel/m

*Adam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/m*&
_output_shapes
:0`*
dtype0

Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*%
shared_nameAdam/conv2d_1/bias/m
y
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
_output_shapes
:0*
dtype0

Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*'
shared_nameAdam/conv2d_1/kernel/m

*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*&
_output_shapes
:0*
dtype0
|
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv2d/bias/m
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d/kernel/m

(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
:*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0

random_rotation/StateVarVarHandleOp*
_output_shapes
: *
dtype0	*
shape:*)
shared_namerandom_rotation/StateVar

,random_rotation/StateVar/Read/ReadVariableOpReadVariableOprandom_rotation/StateVar*
_output_shapes
:*
dtype0	

random_flip/StateVarVarHandleOp*
_output_shapes
: *
dtype0	*
shape:*%
shared_namerandom_flip/StateVar
y
(random_flip/StateVar/Read/ReadVariableOpReadVariableOprandom_flip/StateVar*
_output_shapes
:*
dtype0	
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
à*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
à*
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:`*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:`*
dtype0

conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0`* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
:0`*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:0*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:0*
dtype0

conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:0* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:0*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:*
dtype0

!serving_default_random_flip_inputPlaceholder*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*&
shape:ÿÿÿÿÿÿÿÿÿ
À
StatefulPartitionedCallStatefulPartitionedCall!serving_default_random_flip_inputconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasdense/kernel
dense/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference_signature_wrapper_6691

NoOpNoOp
µ[
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ðZ
valueæZBãZ BÜZ
Ä
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer-4
layer_with_weights-1
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer-9
layer_with_weights-3
layer-10
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
§
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator*
§
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses
"_random_generator*

#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses* 
È
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

/kernel
0bias
 1_jit_compiled_convolution_op*

2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses* 
È
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses

>kernel
?bias
 @_jit_compiled_convolution_op*

A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses* 
È
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses

Mkernel
Nbias
 O_jit_compiled_convolution_op*

P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses* 
¥
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses
\_random_generator* 
¦
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses

ckernel
dbias*
<
/0
01
>2
?3
M4
N5
c6
d7*
<
/0
01
>2
?3
M4
N5
c6
d7*
* 
°
enon_trainable_variables

flayers
gmetrics
hlayer_regularization_losses
ilayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
jtrace_0
ktrace_1
ltrace_2
mtrace_3* 
6
ntrace_0
otrace_1
ptrace_2
qtrace_3* 
* 
ä
riter

sbeta_1

tbeta_2
	udecay
vlearning_rate/mÚ0mÛ>mÜ?mÝMmÞNmßcmàdmá/vâ0vã>vä?våMvæNvçcvèdvé*

wserving_default* 
* 
* 
* 

xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

}trace_0
~trace_1* 

trace_0
trace_1* 


_generator*
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses* 

trace_0
trace_1* 

trace_0
trace_1* 


_generator*
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses* 

trace_0* 

trace_0* 

/0
01*

/0
01*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses*

trace_0* 

trace_0* 
]W
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses* 

trace_0* 

 trace_0* 

>0
?1*

>0
?1*
* 

¡non_trainable_variables
¢layers
£metrics
 ¤layer_regularization_losses
¥layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses*

¦trace_0* 

§trace_0* 
_Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

¨non_trainable_variables
©layers
ªmetrics
 «layer_regularization_losses
¬layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses* 

­trace_0* 

®trace_0* 

M0
N1*

M0
N1*
* 

¯non_trainable_variables
°layers
±metrics
 ²layer_regularization_losses
³layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses*

´trace_0* 

µtrace_0* 
_Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

¶non_trainable_variables
·layers
¸metrics
 ¹layer_regularization_losses
ºlayer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses* 

»trace_0* 

¼trace_0* 
* 
* 
* 

½non_trainable_variables
¾layers
¿metrics
 Àlayer_regularization_losses
Álayer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses* 

Âtrace_0
Ãtrace_1* 

Ätrace_0
Åtrace_1* 
* 

c0
d1*

c0
d1*
* 

Ænon_trainable_variables
Çlayers
Èmetrics
 Élayer_regularization_losses
Êlayer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses*

Ëtrace_0* 

Ìtrace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
R
0
1
2
3
4
5
6
7
	8

9
10*

Í0
Î1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

Ï
_state_var*
* 
* 
* 
* 
* 
* 
* 
* 
* 

Ð
_state_var*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
Ñ	variables
Ò	keras_api

Ótotal

Ôcount*
M
Õ	variables
Ö	keras_api

×total

Øcount
Ù
_fn_kwargs*
xr
VARIABLE_VALUErandom_flip/StateVarJlayer-0/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUErandom_rotation/StateVarJlayer-1/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUE*

Ó0
Ô1*

Ñ	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

×0
Ø1*

Õ	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
z
VARIABLE_VALUEAdam/conv2d/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/conv2d/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/conv2d_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/conv2d_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/conv2d/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUEAdam/conv2d/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/conv2d_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/conv2d_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/conv2d_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/dense/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/dense/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ß
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp(random_flip/StateVar/Read/ReadVariableOp,random_rotation/StateVar/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp*Adam/conv2d_2/kernel/m/Read/ReadVariableOp(Adam/conv2d_2/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp*Adam/conv2d_2/kernel/v/Read/ReadVariableOp(Adam/conv2d_2/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOpConst*0
Tin)
'2%			*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *&
f!R
__inference__traced_save_9083

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/biasdense/kernel
dense/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_raterandom_flip/StateVarrandom_rotation/StateVartotal_1count_1totalcountAdam/conv2d/kernel/mAdam/conv2d/bias/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/mAdam/conv2d_2/kernel/mAdam/conv2d_2/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/conv2d/kernel/vAdam/conv2d/bias/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/vAdam/conv2d_2/kernel/vAdam/conv2d_2/bias/vAdam/dense/kernel/vAdam/dense/bias/v*/
Tin(
&2$*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__traced_restore_9198´¹*


 __inference__traced_restore_9198
file_prefix8
assignvariableop_conv2d_kernel:,
assignvariableop_1_conv2d_bias:<
"assignvariableop_2_conv2d_1_kernel:0.
 assignvariableop_3_conv2d_1_bias:0<
"assignvariableop_4_conv2d_2_kernel:0`.
 assignvariableop_5_conv2d_2_bias:`3
assignvariableop_6_dense_kernel:
à+
assignvariableop_7_dense_bias:&
assignvariableop_8_adam_iter:	 (
assignvariableop_9_adam_beta_1: )
assignvariableop_10_adam_beta_2: (
assignvariableop_11_adam_decay: 0
&assignvariableop_12_adam_learning_rate: 6
(assignvariableop_13_random_flip_statevar:	:
,assignvariableop_14_random_rotation_statevar:	%
assignvariableop_15_total_1: %
assignvariableop_16_count_1: #
assignvariableop_17_total: #
assignvariableop_18_count: B
(assignvariableop_19_adam_conv2d_kernel_m:4
&assignvariableop_20_adam_conv2d_bias_m:D
*assignvariableop_21_adam_conv2d_1_kernel_m:06
(assignvariableop_22_adam_conv2d_1_bias_m:0D
*assignvariableop_23_adam_conv2d_2_kernel_m:0`6
(assignvariableop_24_adam_conv2d_2_bias_m:`;
'assignvariableop_25_adam_dense_kernel_m:
à3
%assignvariableop_26_adam_dense_bias_m:B
(assignvariableop_27_adam_conv2d_kernel_v:4
&assignvariableop_28_adam_conv2d_bias_v:D
*assignvariableop_29_adam_conv2d_1_kernel_v:06
(assignvariableop_30_adam_conv2d_1_bias_v:0D
*assignvariableop_31_adam_conv2d_2_kernel_v:0`6
(assignvariableop_32_adam_conv2d_2_bias_v:`;
'assignvariableop_33_adam_dense_kernel_v:
à3
%assignvariableop_34_adam_dense_bias_v:
identity_36¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Ê
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*ð
valueæBã$B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBJlayer-0/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBJlayer-1/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH¸
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Õ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*¦
_output_shapes
::::::::::::::::::::::::::::::::::::*2
dtypes(
&2$			[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_iterIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_beta_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_beta_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_decayIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp&assignvariableop_12_adam_learning_rateIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_13AssignVariableOp(assignvariableop_13_random_flip_statevarIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_14AssignVariableOp,assignvariableop_14_random_rotation_statevarIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_total_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_count_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOpassignvariableop_17_totalIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOpassignvariableop_18_countIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp(assignvariableop_19_adam_conv2d_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp&assignvariableop_20_adam_conv2d_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp*assignvariableop_21_adam_conv2d_1_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp(assignvariableop_22_adam_conv2d_1_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp*assignvariableop_23_adam_conv2d_2_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_conv2d_2_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp'assignvariableop_25_adam_dense_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp%assignvariableop_26_adam_dense_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp(assignvariableop_27_adam_conv2d_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp&assignvariableop_28_adam_conv2d_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_conv2d_1_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_conv2d_1_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_conv2d_2_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_conv2d_2_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOp'assignvariableop_33_adam_dense_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOp%assignvariableop_34_adam_dense_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Ñ
Identity_35Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_36IdentityIdentity_35:output:0^NoOp_1*
T0*
_output_shapes
: ¾
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_36Identity_36:output:0*[
_input_shapesJ
H: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ã
D
(__inference_rescaling_layer_call_fn_8798

inputs
identity¸
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_rescaling_layer_call_and_return_conditional_losses_5321j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«
»

Cloop_body_transform_ImageProjectiveTransformV3_pfor_while_cond_8724
loop_body_transform_imageprojectivetransformv3_pfor_while_loop_body_transform_imageprojectivetransformv3_pfor_while_loop_counter
loop_body_transform_imageprojectivetransformv3_pfor_while_loop_body_transform_imageprojectivetransformv3_pfor_while_maximum_iterationsI
Eloop_body_transform_imageprojectivetransformv3_pfor_while_placeholderK
Gloop_body_transform_imageprojectivetransformv3_pfor_while_placeholder_1
loop_body_transform_imageprojectivetransformv3_pfor_while_less_loop_body_transform_imageprojectivetransformv3_pfor_strided_slice
loop_body_transform_imageprojectivetransformv3_pfor_while_loop_body_transform_imageprojectivetransformv3_pfor_while_cond_8724___redundant_placeholder0
loop_body_transform_imageprojectivetransformv3_pfor_while_loop_body_transform_imageprojectivetransformv3_pfor_while_cond_8724___redundant_placeholder1
loop_body_transform_imageprojectivetransformv3_pfor_while_loop_body_transform_imageprojectivetransformv3_pfor_while_cond_8724___redundant_placeholder2
loop_body_transform_imageprojectivetransformv3_pfor_while_loop_body_transform_imageprojectivetransformv3_pfor_while_cond_8724___redundant_placeholder3F
Bloop_body_transform_imageprojectivetransformv3_pfor_while_identity
±
>loop_body/transform/ImageProjectiveTransformV3/pfor/while/LessLessEloop_body_transform_imageprojectivetransformv3_pfor_while_placeholderloop_body_transform_imageprojectivetransformv3_pfor_while_less_loop_body_transform_imageprojectivetransformv3_pfor_strided_slice*
T0*
_output_shapes
: ³
Bloop_body/transform/ImageProjectiveTransformV3/pfor/while/IdentityIdentityBloop_body/transform/ImageProjectiveTransformV3/pfor/while/Less:z:0*
T0
*
_output_shapes
: "
Bloop_body_transform_imageprojectivetransformv3_pfor_while_identityKloop_body/transform/ImageProjectiveTransformV3/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
: : : : : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
:
éÂ	

D__inference_sequential_layer_call_and_return_conditional_losses_7792

inputsP
Brandom_rotation_loop_body_stateful_uniform_rngreadandskip_resource:	?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource:06
(conv2d_1_biasadd_readvariableop_resource:0A
'conv2d_2_conv2d_readvariableop_resource:0`6
(conv2d_2_biasadd_readvariableop_resource:`8
$dense_matmul_readvariableop_resource:
à3
%dense_biasadd_readvariableop_resource:
identity¢conv2d/BiasAdd/ReadVariableOp¢conv2d/Conv2D/ReadVariableOp¢conv2d_1/BiasAdd/ReadVariableOp¢conv2d_1/Conv2D/ReadVariableOp¢conv2d_2/BiasAdd/ReadVariableOp¢conv2d_2/Conv2D/ReadVariableOp¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOp¢9random_rotation/loop_body/stateful_uniform/RngReadAndSkip¢Drandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/whileK
random_flip/map/ShapeShapeinputs*
T0*
_output_shapes
:m
#random_flip/map/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%random_flip/map/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%random_flip/map/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¡
random_flip/map/strided_sliceStridedSlicerandom_flip/map/Shape:output:0,random_flip/map/strided_slice/stack:output:0.random_flip/map/strided_slice/stack_1:output:0.random_flip/map/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskv
+random_flip/map/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿâ
random_flip/map/TensorArrayV2TensorListReserve4random_flip/map/TensorArrayV2/element_shape:output:0&random_flip/map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Erandom_flip/map/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ù
7random_flip/map/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorinputsNrandom_flip/map/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒW
random_flip/map/ConstConst*
_output_shapes
: *
dtype0*
value	B : x
-random_flip/map/TensorArrayV2_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿæ
random_flip/map/TensorArrayV2_1TensorListReserve6random_flip/map/TensorArrayV2_1/element_shape:output:0&random_flip/map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒd
"random_flip/map/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : â
random_flip/map/whileStatelessWhile+random_flip/map/while/loop_counter:output:0&random_flip/map/strided_slice:output:0random_flip/map/Const:output:0(random_flip/map/TensorArrayV2_1:handle:0&random_flip/map/strided_slice:output:0Grandom_flip/map/TensorArrayUnstack/TensorListFromTensor:output_handle:0*
T

2*
_lower_using_switch_merge(*
_num_original_outputs* 
_output_shapes
: : : : : : * 
_read_only_resource_inputs
 *
_stateful_parallelism( *+
body#R!
random_flip_map_while_body_6793*+
cond#R!
random_flip_map_while_cond_6792*
output_shapes
: : : : : : 
@random_flip/map/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ø
2random_flip/map/TensorArrayV2Stack/TensorListStackTensorListStackrandom_flip/map/while:output:3Irandom_flip/map/TensorArrayV2Stack/TensorListStack/element_shape:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
random_rotation/ShapeShape;random_flip/map/TensorArrayV2Stack/TensorListStack:tensor:0*
T0*
_output_shapes
:m
#random_rotation/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%random_rotation/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:o
%random_rotation/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¡
random_rotation/strided_sliceStridedSlicerandom_rotation/Shape:output:0,random_rotation/strided_slice/stack:output:0.random_rotation/strided_slice/stack_1:output:0.random_rotation/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masky
random_rotation/Rank/packedPack&random_rotation/strided_slice:output:0*
N*
T0*
_output_shapes
:V
random_rotation/RankConst*
_output_shapes
: *
dtype0*
value	B :]
random_rotation/range/startConst*
_output_shapes
: *
dtype0*
value	B : ]
random_rotation/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :¥
random_rotation/rangeRange$random_rotation/range/start:output:0random_rotation/Rank:output:0$random_rotation/range/delta:output:0*
_output_shapes
:w
random_rotation/Max/inputPack&random_rotation/strided_slice:output:0*
N*
T0*
_output_shapes
:
random_rotation/MaxMax"random_rotation/Max/input:output:0random_rotation/range:output:0*
T0*
_output_shapes
: x
6random_rotation/loop_body/PlaceholderWithDefault/inputConst*
_output_shapes
: *
dtype0*
value	B : ½
0random_rotation/loop_body/PlaceholderWithDefaultPlaceholderWithDefault?random_rotation/loop_body/PlaceholderWithDefault/input:output:0*
_output_shapes
: *
dtype0*
shape: 
random_rotation/loop_body/ShapeShape;random_flip/map/TensorArrayV2Stack/TensorListStack:tensor:0*
T0*
_output_shapes
:w
-random_rotation/loop_body/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/random_rotation/loop_body/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/random_rotation/loop_body/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ó
'random_rotation/loop_body/strided_sliceStridedSlice(random_rotation/loop_body/Shape:output:06random_rotation/loop_body/strided_slice/stack:output:08random_rotation/loop_body/strided_slice/stack_1:output:08random_rotation/loop_body/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#random_rotation/loop_body/Greater/yConst*
_output_shapes
: *
dtype0*
value	B :­
!random_rotation/loop_body/GreaterGreater0random_rotation/loop_body/strided_slice:output:0,random_rotation/loop_body/Greater/y:output:0*
T0*
_output_shapes
: f
$random_rotation/loop_body/SelectV2/eConst*
_output_shapes
: *
dtype0*
value	B : à
"random_rotation/loop_body/SelectV2SelectV2%random_rotation/loop_body/Greater:z:09random_rotation/loop_body/PlaceholderWithDefault:output:0-random_rotation/loop_body/SelectV2/e:output:0*
T0*
_output_shapes
: i
'random_rotation/loop_body/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
"random_rotation/loop_body/GatherV2GatherV2;random_flip/map/TensorArrayV2Stack/TensorListStack:tensor:0+random_rotation/loop_body/SelectV2:output:00random_rotation/loop_body/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*$
_output_shapes
:z
0random_rotation/loop_body/stateful_uniform/shapeConst*
_output_shapes
:*
dtype0*
valueB:s
.random_rotation/loop_body/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|Ù ¿s
.random_rotation/loop_body/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|Ù ?z
0random_rotation/loop_body/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: Î
/random_rotation/loop_body/stateful_uniform/ProdProd9random_rotation/loop_body/stateful_uniform/shape:output:09random_rotation/loop_body/stateful_uniform/Const:output:0*
T0*
_output_shapes
: s
1random_rotation/loop_body/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :£
1random_rotation/loop_body/stateful_uniform/Cast_1Cast8random_rotation/loop_body/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 
9random_rotation/loop_body/stateful_uniform/RngReadAndSkipRngReadAndSkipBrandom_rotation_loop_body_stateful_uniform_rngreadandskip_resource:random_rotation/loop_body/stateful_uniform/Cast/x:output:05random_rotation/loop_body/stateful_uniform/Cast_1:y:0*
_output_shapes
:
>random_rotation/loop_body/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
@random_rotation/loop_body/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
@random_rotation/loop_body/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:®
8random_rotation/loop_body/stateful_uniform/strided_sliceStridedSliceArandom_rotation/loop_body/stateful_uniform/RngReadAndSkip:value:0Grandom_rotation/loop_body/stateful_uniform/strided_slice/stack:output:0Irandom_rotation/loop_body/stateful_uniform/strided_slice/stack_1:output:0Irandom_rotation/loop_body/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask±
2random_rotation/loop_body/stateful_uniform/BitcastBitcastArandom_rotation/loop_body/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
@random_rotation/loop_body/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Brandom_rotation/loop_body/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Brandom_rotation/loop_body/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¤
:random_rotation/loop_body/stateful_uniform/strided_slice_1StridedSliceArandom_rotation/loop_body/stateful_uniform/RngReadAndSkip:value:0Irandom_rotation/loop_body/stateful_uniform/strided_slice_1/stack:output:0Krandom_rotation/loop_body/stateful_uniform/strided_slice_1/stack_1:output:0Krandom_rotation/loop_body/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:µ
4random_rotation/loop_body/stateful_uniform/Bitcast_1BitcastCrandom_rotation/loop_body/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0
Grandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :
Crandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV29random_rotation/loop_body/stateful_uniform/shape:output:0=random_rotation/loop_body/stateful_uniform/Bitcast_1:output:0;random_rotation/loop_body/stateful_uniform/Bitcast:output:0Prandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/alg:output:0*
_output_shapes
:È
.random_rotation/loop_body/stateful_uniform/subSub7random_rotation/loop_body/stateful_uniform/max:output:07random_rotation/loop_body/stateful_uniform/min:output:0*
T0*
_output_shapes
: Ü
.random_rotation/loop_body/stateful_uniform/mulMulLrandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2:output:02random_rotation/loop_body/stateful_uniform/sub:z:0*
T0*
_output_shapes
:Å
*random_rotation/loop_body/stateful_uniformAddV22random_rotation/loop_body/stateful_uniform/mul:z:07random_rotation/loop_body/stateful_uniform/min:output:0*
T0*
_output_shapes
:j
(random_rotation/loop_body/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : Å
$random_rotation/loop_body/ExpandDims
ExpandDims+random_rotation/loop_body/GatherV2:output:01random_rotation/loop_body/ExpandDims/dim:output:0*
T0*(
_output_shapes
:z
!random_rotation/loop_body/Shape_1Const*
_output_shapes
:*
dtype0*%
valueB"            
/random_rotation/loop_body/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿ
1random_rotation/loop_body/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ{
1random_rotation/loop_body/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ý
)random_rotation/loop_body/strided_slice_1StridedSlice*random_rotation/loop_body/Shape_1:output:08random_rotation/loop_body/strided_slice_1/stack:output:0:random_rotation/loop_body/strided_slice_1/stack_1:output:0:random_rotation/loop_body/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
random_rotation/loop_body/CastCast2random_rotation/loop_body/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 
/random_rotation/loop_body/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿ
1random_rotation/loop_body/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ{
1random_rotation/loop_body/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ý
)random_rotation/loop_body/strided_slice_2StridedSlice*random_rotation/loop_body/Shape_1:output:08random_rotation/loop_body/strided_slice_2/stack:output:0:random_rotation/loop_body/strided_slice_2/stack_1:output:0:random_rotation/loop_body/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
 random_rotation/loop_body/Cast_1Cast2random_rotation/loop_body/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: t
/random_rotation/loop_body/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?µ
-random_rotation/loop_body/rotation_matrix/subSub$random_rotation/loop_body/Cast_1:y:08random_rotation/loop_body/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: 
-random_rotation/loop_body/rotation_matrix/CosCos.random_rotation/loop_body/stateful_uniform:z:0*
T0*
_output_shapes
:v
1random_rotation/loop_body/rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¹
/random_rotation/loop_body/rotation_matrix/sub_1Sub$random_rotation/loop_body/Cast_1:y:0:random_rotation/loop_body/rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: Á
-random_rotation/loop_body/rotation_matrix/mulMul1random_rotation/loop_body/rotation_matrix/Cos:y:03random_rotation/loop_body/rotation_matrix/sub_1:z:0*
T0*
_output_shapes
:
-random_rotation/loop_body/rotation_matrix/SinSin.random_rotation/loop_body/stateful_uniform:z:0*
T0*
_output_shapes
:v
1random_rotation/loop_body/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?·
/random_rotation/loop_body/rotation_matrix/sub_2Sub"random_rotation/loop_body/Cast:y:0:random_rotation/loop_body/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: Ã
/random_rotation/loop_body/rotation_matrix/mul_1Mul1random_rotation/loop_body/rotation_matrix/Sin:y:03random_rotation/loop_body/rotation_matrix/sub_2:z:0*
T0*
_output_shapes
:Ã
/random_rotation/loop_body/rotation_matrix/sub_3Sub1random_rotation/loop_body/rotation_matrix/mul:z:03random_rotation/loop_body/rotation_matrix/mul_1:z:0*
T0*
_output_shapes
:Ã
/random_rotation/loop_body/rotation_matrix/sub_4Sub1random_rotation/loop_body/rotation_matrix/sub:z:03random_rotation/loop_body/rotation_matrix/sub_3:z:0*
T0*
_output_shapes
:x
3random_rotation/loop_body/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @Ô
1random_rotation/loop_body/rotation_matrix/truedivRealDiv3random_rotation/loop_body/rotation_matrix/sub_4:z:0<random_rotation/loop_body/rotation_matrix/truediv/y:output:0*
T0*
_output_shapes
:v
1random_rotation/loop_body/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?·
/random_rotation/loop_body/rotation_matrix/sub_5Sub"random_rotation/loop_body/Cast:y:0:random_rotation/loop_body/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 
/random_rotation/loop_body/rotation_matrix/Sin_1Sin.random_rotation/loop_body/stateful_uniform:z:0*
T0*
_output_shapes
:v
1random_rotation/loop_body/rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¹
/random_rotation/loop_body/rotation_matrix/sub_6Sub$random_rotation/loop_body/Cast_1:y:0:random_rotation/loop_body/rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: Å
/random_rotation/loop_body/rotation_matrix/mul_2Mul3random_rotation/loop_body/rotation_matrix/Sin_1:y:03random_rotation/loop_body/rotation_matrix/sub_6:z:0*
T0*
_output_shapes
:
/random_rotation/loop_body/rotation_matrix/Cos_1Cos.random_rotation/loop_body/stateful_uniform:z:0*
T0*
_output_shapes
:v
1random_rotation/loop_body/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?·
/random_rotation/loop_body/rotation_matrix/sub_7Sub"random_rotation/loop_body/Cast:y:0:random_rotation/loop_body/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: Å
/random_rotation/loop_body/rotation_matrix/mul_3Mul3random_rotation/loop_body/rotation_matrix/Cos_1:y:03random_rotation/loop_body/rotation_matrix/sub_7:z:0*
T0*
_output_shapes
:Å
-random_rotation/loop_body/rotation_matrix/addAddV23random_rotation/loop_body/rotation_matrix/mul_2:z:03random_rotation/loop_body/rotation_matrix/mul_3:z:0*
T0*
_output_shapes
:Ã
/random_rotation/loop_body/rotation_matrix/sub_8Sub3random_rotation/loop_body/rotation_matrix/sub_5:z:01random_rotation/loop_body/rotation_matrix/add:z:0*
T0*
_output_shapes
:z
5random_rotation/loop_body/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @Ø
3random_rotation/loop_body/rotation_matrix/truediv_1RealDiv3random_rotation/loop_body/rotation_matrix/sub_8:z:0>random_rotation/loop_body/rotation_matrix/truediv_1/y:output:0*
T0*
_output_shapes
:y
/random_rotation/loop_body/rotation_matrix/ShapeConst*
_output_shapes
:*
dtype0*
valueB:
=random_rotation/loop_body/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?random_rotation/loop_body/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?random_rotation/loop_body/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:£
7random_rotation/loop_body/rotation_matrix/strided_sliceStridedSlice8random_rotation/loop_body/rotation_matrix/Shape:output:0Frandom_rotation/loop_body/rotation_matrix/strided_slice/stack:output:0Hrandom_rotation/loop_body/rotation_matrix/strided_slice/stack_1:output:0Hrandom_rotation/loop_body/rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
/random_rotation/loop_body/rotation_matrix/Cos_2Cos.random_rotation/loop_body/stateful_uniform:z:0*
T0*
_output_shapes
:
?random_rotation/loop_body/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        
Arandom_rotation/loop_body/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
Arandom_rotation/loop_body/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Í
9random_rotation/loop_body/rotation_matrix/strided_slice_1StridedSlice3random_rotation/loop_body/rotation_matrix/Cos_2:y:0Hrandom_rotation/loop_body/rotation_matrix/strided_slice_1/stack:output:0Jrandom_rotation/loop_body/rotation_matrix/strided_slice_1/stack_1:output:0Jrandom_rotation/loop_body/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask*
new_axis_mask
/random_rotation/loop_body/rotation_matrix/Sin_2Sin.random_rotation/loop_body/stateful_uniform:z:0*
T0*
_output_shapes
:
?random_rotation/loop_body/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        
Arandom_rotation/loop_body/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
Arandom_rotation/loop_body/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Í
9random_rotation/loop_body/rotation_matrix/strided_slice_2StridedSlice3random_rotation/loop_body/rotation_matrix/Sin_2:y:0Hrandom_rotation/loop_body/rotation_matrix/strided_slice_2/stack:output:0Jrandom_rotation/loop_body/rotation_matrix/strided_slice_2/stack_1:output:0Jrandom_rotation/loop_body/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask*
new_axis_mask¡
-random_rotation/loop_body/rotation_matrix/NegNegBrandom_rotation/loop_body/rotation_matrix/strided_slice_2:output:0*
T0*
_output_shapes

:
?random_rotation/loop_body/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        
Arandom_rotation/loop_body/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
Arandom_rotation/loop_body/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ï
9random_rotation/loop_body/rotation_matrix/strided_slice_3StridedSlice5random_rotation/loop_body/rotation_matrix/truediv:z:0Hrandom_rotation/loop_body/rotation_matrix/strided_slice_3/stack:output:0Jrandom_rotation/loop_body/rotation_matrix/strided_slice_3/stack_1:output:0Jrandom_rotation/loop_body/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask*
new_axis_mask
/random_rotation/loop_body/rotation_matrix/Sin_3Sin.random_rotation/loop_body/stateful_uniform:z:0*
T0*
_output_shapes
:
?random_rotation/loop_body/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        
Arandom_rotation/loop_body/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
Arandom_rotation/loop_body/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Í
9random_rotation/loop_body/rotation_matrix/strided_slice_4StridedSlice3random_rotation/loop_body/rotation_matrix/Sin_3:y:0Hrandom_rotation/loop_body/rotation_matrix/strided_slice_4/stack:output:0Jrandom_rotation/loop_body/rotation_matrix/strided_slice_4/stack_1:output:0Jrandom_rotation/loop_body/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask*
new_axis_mask
/random_rotation/loop_body/rotation_matrix/Cos_3Cos.random_rotation/loop_body/stateful_uniform:z:0*
T0*
_output_shapes
:
?random_rotation/loop_body/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        
Arandom_rotation/loop_body/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
Arandom_rotation/loop_body/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Í
9random_rotation/loop_body/rotation_matrix/strided_slice_5StridedSlice3random_rotation/loop_body/rotation_matrix/Cos_3:y:0Hrandom_rotation/loop_body/rotation_matrix/strided_slice_5/stack:output:0Jrandom_rotation/loop_body/rotation_matrix/strided_slice_5/stack_1:output:0Jrandom_rotation/loop_body/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask*
new_axis_mask
?random_rotation/loop_body/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        
Arandom_rotation/loop_body/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
Arandom_rotation/loop_body/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ñ
9random_rotation/loop_body/rotation_matrix/strided_slice_6StridedSlice7random_rotation/loop_body/rotation_matrix/truediv_1:z:0Hrandom_rotation/loop_body/rotation_matrix/strided_slice_6/stack:output:0Jrandom_rotation/loop_body/rotation_matrix/strided_slice_6/stack_1:output:0Jrandom_rotation/loop_body/rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask*
new_axis_maskz
8random_rotation/loop_body/rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ñ
6random_rotation/loop_body/rotation_matrix/zeros/packedPack@random_rotation/loop_body/rotation_matrix/strided_slice:output:0Arandom_rotation/loop_body/rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:z
5random_rotation/loop_body/rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    á
/random_rotation/loop_body/rotation_matrix/zerosFill?random_rotation/loop_body/rotation_matrix/zeros/packed:output:0>random_rotation/loop_body/rotation_matrix/zeros/Const:output:0*
T0*
_output_shapes

:w
5random_rotation/loop_body/rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ï
0random_rotation/loop_body/rotation_matrix/concatConcatV2Brandom_rotation/loop_body/rotation_matrix/strided_slice_1:output:01random_rotation/loop_body/rotation_matrix/Neg:y:0Brandom_rotation/loop_body/rotation_matrix/strided_slice_3:output:0Brandom_rotation/loop_body/rotation_matrix/strided_slice_4:output:0Brandom_rotation/loop_body/rotation_matrix/strided_slice_5:output:0Brandom_rotation/loop_body/rotation_matrix/strided_slice_6:output:08random_rotation/loop_body/rotation_matrix/zeros:output:0>random_rotation/loop_body/rotation_matrix/concat/axis:output:0*
N*
T0*
_output_shapes

:
)random_rotation/loop_body/transform/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            
7random_rotation/loop_body/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:
9random_rotation/loop_body/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9random_rotation/loop_body/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ñ
1random_rotation/loop_body/transform/strided_sliceStridedSlice2random_rotation/loop_body/transform/Shape:output:0@random_rotation/loop_body/transform/strided_slice/stack:output:0Brandom_rotation/loop_body/transform/strided_slice/stack_1:output:0Brandom_rotation/loop_body/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:s
.random_rotation/loop_body/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    §
>random_rotation/loop_body/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3-random_rotation/loop_body/ExpandDims:output:09random_rotation/loop_body/rotation_matrix/concat:output:0:random_rotation/loop_body/transform/strided_slice:output:07random_rotation/loop_body/transform/fill_value:output:0*(
_output_shapes
:*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEARÇ
!random_rotation/loop_body/SqueezeSqueezeSrandom_rotation/loop_body/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*$
_output_shapes
:*
squeeze_dims
 l
"random_rotation/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
random_rotation/pfor/ReshapeReshaperandom_rotation/Max:output:0+random_rotation/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:b
 random_rotation/pfor/range/startConst*
_output_shapes
: *
dtype0*
value	B : b
 random_rotation/pfor/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :¼
random_rotation/pfor/rangeRange)random_rotation/pfor/range/start:output:0random_rotation/Max:output:0)random_rotation/pfor/range/delta:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Rrandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Trandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Trandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ä
Lrandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/strided_sliceStridedSlice%random_rotation/pfor/Reshape:output:0[random_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/strided_slice/stack:output:0]random_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/strided_slice/stack_1:output:0]random_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask¥
Zrandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿï
Lrandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/TensorArrayV2TensorListReservecrandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/TensorArrayV2/element_shape:output:0Urandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0	*

shape_type0:éèÐ
Drandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : ¢
Wrandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Qrandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 	
Drandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/whileWhileZrandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/while/loop_counter:output:0`random_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/while/maximum_iterations:output:0Mrandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/Const:output:0Urandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/TensorArrayV2:handle:0Urandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/strided_slice:output:0Brandom_rotation_loop_body_stateful_uniform_rngreadandskip_resource:random_rotation/loop_body/stateful_uniform/Cast/x:output:05random_rotation/loop_body/stateful_uniform/Cast_1:y:0:^random_rotation/loop_body/stateful_uniform/RngReadAndSkip*
T

2*
_lower_using_switch_merge(*
_num_original_outputs*$
_output_shapes
: : : : : : : : * 
_read_only_resource_inputs
 *
_stateful_parallelism( *Z
bodyRRP
Nrandom_rotation_loop_body_stateful_uniform_RngReadAndSkip_pfor_while_body_6989*Z
condRRP
Nrandom_rotation_loop_body_stateful_uniform_RngReadAndSkip_pfor_while_cond_6988*#
output_shapes
: : : : : : : : 
Frandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 °
_random_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ð
Qrandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/TensorListConcatV2TensorListConcatV2Mrandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/while:output:3hrandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/TensorListConcatV2/element_shape:output:0Orandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/Const_1:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0	*

shape_type0
Mrandom_rotation/loop_body/stateful_uniform/strided_slice/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: 
Irandom_rotation/loop_body/stateful_uniform/strided_slice/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ó
Drandom_rotation/loop_body/stateful_uniform/strided_slice/pfor/concatConcatV2Vrandom_rotation/loop_body/stateful_uniform/strided_slice/pfor/concat/values_0:output:0Grandom_rotation/loop_body/stateful_uniform/strided_slice/stack:output:0Rrandom_rotation/loop_body/stateful_uniform/strided_slice/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
Orandom_rotation/loop_body/stateful_uniform/strided_slice/pfor/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB: 
Krandom_rotation/loop_body/stateful_uniform/strided_slice/pfor/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : û
Frandom_rotation/loop_body/stateful_uniform/strided_slice/pfor/concat_1ConcatV2Xrandom_rotation/loop_body/stateful_uniform/strided_slice/pfor/concat_1/values_0:output:0Irandom_rotation/loop_body/stateful_uniform/strided_slice/stack_1:output:0Trandom_rotation/loop_body/stateful_uniform/strided_slice/pfor/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
Orandom_rotation/loop_body/stateful_uniform/strided_slice/pfor/concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB:
Krandom_rotation/loop_body/stateful_uniform/strided_slice/pfor/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : û
Frandom_rotation/loop_body/stateful_uniform/strided_slice/pfor/concat_2ConcatV2Xrandom_rotation/loop_body/stateful_uniform/strided_slice/pfor/concat_2/values_0:output:0Irandom_rotation/loop_body/stateful_uniform/strided_slice/stack_2:output:0Trandom_rotation/loop_body/stateful_uniform/strided_slice/pfor/concat_2/axis:output:0*
N*
T0*
_output_shapes
:
Jrandom_rotation/loop_body/stateful_uniform/strided_slice/pfor/StridedSliceStridedSliceZrandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/TensorListConcatV2:tensor:0Mrandom_rotation/loop_body/stateful_uniform/strided_slice/pfor/concat:output:0Orandom_rotation/loop_body/stateful_uniform/strided_slice/pfor/concat_1:output:0Orandom_rotation/loop_body/stateful_uniform/strided_slice/pfor/concat_2:output:0*
Index0*
T0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask
Krandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Mrandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Mrandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:È
Erandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/strided_sliceStridedSlice%random_rotation/pfor/Reshape:output:0Trandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/strided_slice/stack:output:0Vrandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/strided_slice/stack_1:output:0Vrandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Srandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÚ
Erandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/TensorArrayV2TensorListReserve\random_rotation/loop_body/stateful_uniform/Bitcast/pfor/TensorArrayV2/element_shape:output:0Nrandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÌ
=random_rotation/loop_body/stateful_uniform/Bitcast/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
Prandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Jrandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Ù
=random_rotation/loop_body/stateful_uniform/Bitcast/pfor/whileStatelessWhileSrandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/while/loop_counter:output:0Yrandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/while/maximum_iterations:output:0Frandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/Const:output:0Nrandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/TensorArrayV2:handle:0Nrandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/strided_slice:output:0Srandom_rotation/loop_body/stateful_uniform/strided_slice/pfor/StridedSlice:output:0*
T

2	*
_lower_using_switch_merge(*
_num_original_outputs*1
_output_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *
_stateful_parallelism( *S
bodyKRI
Grandom_rotation_loop_body_stateful_uniform_Bitcast_pfor_while_body_7054*S
condKRI
Grandom_rotation_loop_body_stateful_uniform_Bitcast_pfor_while_cond_7053*0
output_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ
?random_rotation/loop_body/stateful_uniform/Bitcast/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 ©
Xrandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ´
Jrandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/TensorListConcatV2TensorListConcatV2Frandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/while:output:3arandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/TensorListConcatV2/element_shape:output:0Hrandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/Const_1:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0
Orandom_rotation/loop_body/stateful_uniform/strided_slice_1/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: 
Krandom_rotation/loop_body/stateful_uniform/strided_slice_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : û
Frandom_rotation/loop_body/stateful_uniform/strided_slice_1/pfor/concatConcatV2Xrandom_rotation/loop_body/stateful_uniform/strided_slice_1/pfor/concat/values_0:output:0Irandom_rotation/loop_body/stateful_uniform/strided_slice_1/stack:output:0Trandom_rotation/loop_body/stateful_uniform/strided_slice_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
Qrandom_rotation/loop_body/stateful_uniform/strided_slice_1/pfor/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB: 
Mrandom_rotation/loop_body/stateful_uniform/strided_slice_1/pfor/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Hrandom_rotation/loop_body/stateful_uniform/strided_slice_1/pfor/concat_1ConcatV2Zrandom_rotation/loop_body/stateful_uniform/strided_slice_1/pfor/concat_1/values_0:output:0Krandom_rotation/loop_body/stateful_uniform/strided_slice_1/stack_1:output:0Vrandom_rotation/loop_body/stateful_uniform/strided_slice_1/pfor/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
Qrandom_rotation/loop_body/stateful_uniform/strided_slice_1/pfor/concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB:
Mrandom_rotation/loop_body/stateful_uniform/strided_slice_1/pfor/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
Hrandom_rotation/loop_body/stateful_uniform/strided_slice_1/pfor/concat_2ConcatV2Zrandom_rotation/loop_body/stateful_uniform/strided_slice_1/pfor/concat_2/values_0:output:0Krandom_rotation/loop_body/stateful_uniform/strided_slice_1/stack_2:output:0Vrandom_rotation/loop_body/stateful_uniform/strided_slice_1/pfor/concat_2/axis:output:0*
N*
T0*
_output_shapes
:
Lrandom_rotation/loop_body/stateful_uniform/strided_slice_1/pfor/StridedSliceStridedSliceZrandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/TensorListConcatV2:tensor:0Orandom_rotation/loop_body/stateful_uniform/strided_slice_1/pfor/concat:output:0Qrandom_rotation/loop_body/stateful_uniform/strided_slice_1/pfor/concat_1:output:0Qrandom_rotation/loop_body/stateful_uniform/strided_slice_1/pfor/concat_2:output:0*
Index0*
T0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask
Mrandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Orandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Orandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ð
Grandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/strided_sliceStridedSlice%random_rotation/pfor/Reshape:output:0Vrandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/strided_slice/stack:output:0Xrandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/strided_slice/stack_1:output:0Xrandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask 
Urandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿà
Grandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/TensorArrayV2TensorListReserve^random_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/TensorArrayV2/element_shape:output:0Prandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÌ
?random_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
Rrandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Lrandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ë
?random_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/whileStatelessWhileUrandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/while/loop_counter:output:0[random_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/while/maximum_iterations:output:0Hrandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/Const:output:0Prandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/TensorArrayV2:handle:0Prandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/strided_slice:output:0Urandom_rotation/loop_body/stateful_uniform/strided_slice_1/pfor/StridedSlice:output:0*
T

2	*
_lower_using_switch_merge(*
_num_original_outputs*1
_output_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *
_stateful_parallelism( *U
bodyMRK
Irandom_rotation_loop_body_stateful_uniform_Bitcast_1_pfor_while_body_7121*U
condMRK
Irandom_rotation_loop_body_stateful_uniform_Bitcast_1_pfor_while_cond_7120*0
output_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ
Arandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 «
Zrandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¼
Lrandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/TensorListConcatV2TensorListConcatV2Hrandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/while:output:3crandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/TensorListConcatV2/element_shape:output:0Jrandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/Const_1:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0¦
\random_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: ¨
^random_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:¨
^random_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
Vrandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/strided_sliceStridedSlice%random_rotation/pfor/Reshape:output:0erandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/strided_slice/stack:output:0grandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/strided_slice/stack_1:output:0grandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask¯
drandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Vrandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/TensorArrayV2TensorListReservemrandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/TensorArrayV2/element_shape:output:0_random_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Nrandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : ¬
arandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
[random_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : þ

Nrandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/whileStatelessWhiledrandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/loop_counter:output:0jrandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/maximum_iterations:output:0Wrandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/Const:output:0_random_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/TensorArrayV2:handle:0_random_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/strided_slice:output:0Urandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/TensorListConcatV2:tensor:0Srandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/TensorListConcatV2:tensor:09random_rotation/loop_body/stateful_uniform/shape:output:0Prandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/alg:output:0*
T
2	*
_lower_using_switch_merge(*
_num_original_outputs	*L
_output_shapes:
8: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: * 
_read_only_resource_inputs
 *
_stateful_parallelism( *d
body\RZ
Xrandom_rotation_loop_body_stateful_uniform_StatelessRandomUniformV2_pfor_while_body_7178*d
cond\RZ
Xrandom_rotation_loop_body_stateful_uniform_StatelessRandomUniformV2_pfor_while_cond_7177*K
output_shapes:
8: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: 
Prandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 º
irandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿø
[random_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/TensorListConcatV2TensorListConcatV2Wrandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while:output:3rrandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/TensorListConcatV2/element_shape:output:0Yrandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/Const_1:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0z
8random_rotation/loop_body/stateful_uniform/mul/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :|
:random_rotation/loop_body/stateful_uniform/mul/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : {
9random_rotation/loop_body/stateful_uniform/mul/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :ê
7random_rotation/loop_body/stateful_uniform/mul/pfor/addAddV2Crandom_rotation/loop_body/stateful_uniform/mul/pfor/Rank_1:output:0Brandom_rotation/loop_body/stateful_uniform/mul/pfor/add/y:output:0*
T0*
_output_shapes
: ç
;random_rotation/loop_body/stateful_uniform/mul/pfor/MaximumMaximum;random_rotation/loop_body/stateful_uniform/mul/pfor/add:z:0Arandom_rotation/loop_body/stateful_uniform/mul/pfor/Rank:output:0*
T0*
_output_shapes
: Í
9random_rotation/loop_body/stateful_uniform/mul/pfor/ShapeShapedrandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/TensorListConcatV2:tensor:0*
T0*
_output_shapes
:ã
7random_rotation/loop_body/stateful_uniform/mul/pfor/subSub?random_rotation/loop_body/stateful_uniform/mul/pfor/Maximum:z:0Arandom_rotation/loop_body/stateful_uniform/mul/pfor/Rank:output:0*
T0*
_output_shapes
: 
Arandom_rotation/loop_body/stateful_uniform/mul/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:ô
;random_rotation/loop_body/stateful_uniform/mul/pfor/ReshapeReshape;random_rotation/loop_body/stateful_uniform/mul/pfor/sub:z:0Jrandom_rotation/loop_body/stateful_uniform/mul/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:
>random_rotation/loop_body/stateful_uniform/mul/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:ò
8random_rotation/loop_body/stateful_uniform/mul/pfor/TileTileGrandom_rotation/loop_body/stateful_uniform/mul/pfor/Tile/input:output:0Drandom_rotation/loop_body/stateful_uniform/mul/pfor/Reshape:output:0*
T0*
_output_shapes
: 
Grandom_rotation/loop_body/stateful_uniform/mul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Irandom_rotation/loop_body/stateful_uniform/mul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Irandom_rotation/loop_body/stateful_uniform/mul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ó
Arandom_rotation/loop_body/stateful_uniform/mul/pfor/strided_sliceStridedSliceBrandom_rotation/loop_body/stateful_uniform/mul/pfor/Shape:output:0Prandom_rotation/loop_body/stateful_uniform/mul/pfor/strided_slice/stack:output:0Rrandom_rotation/loop_body/stateful_uniform/mul/pfor/strided_slice/stack_1:output:0Rrandom_rotation/loop_body/stateful_uniform/mul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
Irandom_rotation/loop_body/stateful_uniform/mul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Krandom_rotation/loop_body/stateful_uniform/mul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Krandom_rotation/loop_body/stateful_uniform/mul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
Crandom_rotation/loop_body/stateful_uniform/mul/pfor/strided_slice_1StridedSliceBrandom_rotation/loop_body/stateful_uniform/mul/pfor/Shape:output:0Rrandom_rotation/loop_body/stateful_uniform/mul/pfor/strided_slice_1/stack:output:0Trandom_rotation/loop_body/stateful_uniform/mul/pfor/strided_slice_1/stack_1:output:0Trandom_rotation/loop_body/stateful_uniform/mul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask
?random_rotation/loop_body/stateful_uniform/mul/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
:random_rotation/loop_body/stateful_uniform/mul/pfor/concatConcatV2Jrandom_rotation/loop_body/stateful_uniform/mul/pfor/strided_slice:output:0Arandom_rotation/loop_body/stateful_uniform/mul/pfor/Tile:output:0Lrandom_rotation/loop_body/stateful_uniform/mul/pfor/strided_slice_1:output:0Hrandom_rotation/loop_body/stateful_uniform/mul/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:¥
=random_rotation/loop_body/stateful_uniform/mul/pfor/Reshape_1Reshapedrandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/TensorListConcatV2:tensor:0Crandom_rotation/loop_body/stateful_uniform/mul/pfor/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿì
7random_rotation/loop_body/stateful_uniform/mul/pfor/MulMulFrandom_rotation/loop_body/stateful_uniform/mul/pfor/Reshape_1:output:02random_rotation/loop_body/stateful_uniform/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
4random_rotation/loop_body/stateful_uniform/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :x
6random_rotation/loop_body/stateful_uniform/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : w
5random_rotation/loop_body/stateful_uniform/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :Þ
3random_rotation/loop_body/stateful_uniform/pfor/addAddV2?random_rotation/loop_body/stateful_uniform/pfor/Rank_1:output:0>random_rotation/loop_body/stateful_uniform/pfor/add/y:output:0*
T0*
_output_shapes
: Û
7random_rotation/loop_body/stateful_uniform/pfor/MaximumMaximum7random_rotation/loop_body/stateful_uniform/pfor/add:z:0=random_rotation/loop_body/stateful_uniform/pfor/Rank:output:0*
T0*
_output_shapes
:  
5random_rotation/loop_body/stateful_uniform/pfor/ShapeShape;random_rotation/loop_body/stateful_uniform/mul/pfor/Mul:z:0*
T0*
_output_shapes
:×
3random_rotation/loop_body/stateful_uniform/pfor/subSub;random_rotation/loop_body/stateful_uniform/pfor/Maximum:z:0=random_rotation/loop_body/stateful_uniform/pfor/Rank:output:0*
T0*
_output_shapes
: 
=random_rotation/loop_body/stateful_uniform/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:è
7random_rotation/loop_body/stateful_uniform/pfor/ReshapeReshape7random_rotation/loop_body/stateful_uniform/pfor/sub:z:0Frandom_rotation/loop_body/stateful_uniform/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:
:random_rotation/loop_body/stateful_uniform/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:æ
4random_rotation/loop_body/stateful_uniform/pfor/TileTileCrandom_rotation/loop_body/stateful_uniform/pfor/Tile/input:output:0@random_rotation/loop_body/stateful_uniform/pfor/Reshape:output:0*
T0*
_output_shapes
: 
Crandom_rotation/loop_body/stateful_uniform/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Erandom_rotation/loop_body/stateful_uniform/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Erandom_rotation/loop_body/stateful_uniform/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¿
=random_rotation/loop_body/stateful_uniform/pfor/strided_sliceStridedSlice>random_rotation/loop_body/stateful_uniform/pfor/Shape:output:0Lrandom_rotation/loop_body/stateful_uniform/pfor/strided_slice/stack:output:0Nrandom_rotation/loop_body/stateful_uniform/pfor/strided_slice/stack_1:output:0Nrandom_rotation/loop_body/stateful_uniform/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
Erandom_rotation/loop_body/stateful_uniform/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Grandom_rotation/loop_body/stateful_uniform/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Grandom_rotation/loop_body/stateful_uniform/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Å
?random_rotation/loop_body/stateful_uniform/pfor/strided_slice_1StridedSlice>random_rotation/loop_body/stateful_uniform/pfor/Shape:output:0Nrandom_rotation/loop_body/stateful_uniform/pfor/strided_slice_1/stack:output:0Prandom_rotation/loop_body/stateful_uniform/pfor/strided_slice_1/stack_1:output:0Prandom_rotation/loop_body/stateful_uniform/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask}
;random_rotation/loop_body/stateful_uniform/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
6random_rotation/loop_body/stateful_uniform/pfor/concatConcatV2Frandom_rotation/loop_body/stateful_uniform/pfor/strided_slice:output:0=random_rotation/loop_body/stateful_uniform/pfor/Tile:output:0Hrandom_rotation/loop_body/stateful_uniform/pfor/strided_slice_1:output:0Drandom_rotation/loop_body/stateful_uniform/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:ô
9random_rotation/loop_body/stateful_uniform/pfor/Reshape_1Reshape;random_rotation/loop_body/stateful_uniform/mul/pfor/Mul:z:0?random_rotation/loop_body/stateful_uniform/pfor/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿí
5random_rotation/loop_body/stateful_uniform/pfor/AddV2AddV2Brandom_rotation/loop_body/stateful_uniform/pfor/Reshape_1:output:07random_rotation/loop_body/stateful_uniform/min:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
8random_rotation/loop_body/rotation_matrix/Cos_1/pfor/CosCos9random_rotation/loop_body/stateful_uniform/pfor/AddV2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
9random_rotation/loop_body/rotation_matrix/mul_3/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :}
;random_rotation/loop_body/rotation_matrix/mul_3/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : |
:random_rotation/loop_body/rotation_matrix/mul_3/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :í
8random_rotation/loop_body/rotation_matrix/mul_3/pfor/addAddV2Drandom_rotation/loop_body/rotation_matrix/mul_3/pfor/Rank_1:output:0Crandom_rotation/loop_body/rotation_matrix/mul_3/pfor/add/y:output:0*
T0*
_output_shapes
: ê
<random_rotation/loop_body/rotation_matrix/mul_3/pfor/MaximumMaximum<random_rotation/loop_body/rotation_matrix/mul_3/pfor/add:z:0Brandom_rotation/loop_body/rotation_matrix/mul_3/pfor/Rank:output:0*
T0*
_output_shapes
: ¦
:random_rotation/loop_body/rotation_matrix/mul_3/pfor/ShapeShape<random_rotation/loop_body/rotation_matrix/Cos_1/pfor/Cos:y:0*
T0*
_output_shapes
:æ
8random_rotation/loop_body/rotation_matrix/mul_3/pfor/subSub@random_rotation/loop_body/rotation_matrix/mul_3/pfor/Maximum:z:0Brandom_rotation/loop_body/rotation_matrix/mul_3/pfor/Rank:output:0*
T0*
_output_shapes
: 
Brandom_rotation/loop_body/rotation_matrix/mul_3/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:÷
<random_rotation/loop_body/rotation_matrix/mul_3/pfor/ReshapeReshape<random_rotation/loop_body/rotation_matrix/mul_3/pfor/sub:z:0Krandom_rotation/loop_body/rotation_matrix/mul_3/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:
?random_rotation/loop_body/rotation_matrix/mul_3/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:õ
9random_rotation/loop_body/rotation_matrix/mul_3/pfor/TileTileHrandom_rotation/loop_body/rotation_matrix/mul_3/pfor/Tile/input:output:0Erandom_rotation/loop_body/rotation_matrix/mul_3/pfor/Reshape:output:0*
T0*
_output_shapes
: 
Hrandom_rotation/loop_body/rotation_matrix/mul_3/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Jrandom_rotation/loop_body/rotation_matrix/mul_3/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Jrandom_rotation/loop_body/rotation_matrix/mul_3/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ø
Brandom_rotation/loop_body/rotation_matrix/mul_3/pfor/strided_sliceStridedSliceCrandom_rotation/loop_body/rotation_matrix/mul_3/pfor/Shape:output:0Qrandom_rotation/loop_body/rotation_matrix/mul_3/pfor/strided_slice/stack:output:0Srandom_rotation/loop_body/rotation_matrix/mul_3/pfor/strided_slice/stack_1:output:0Srandom_rotation/loop_body/rotation_matrix/mul_3/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
Jrandom_rotation/loop_body/rotation_matrix/mul_3/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Lrandom_rotation/loop_body/rotation_matrix/mul_3/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Lrandom_rotation/loop_body/rotation_matrix/mul_3/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Þ
Drandom_rotation/loop_body/rotation_matrix/mul_3/pfor/strided_slice_1StridedSliceCrandom_rotation/loop_body/rotation_matrix/mul_3/pfor/Shape:output:0Srandom_rotation/loop_body/rotation_matrix/mul_3/pfor/strided_slice_1/stack:output:0Urandom_rotation/loop_body/rotation_matrix/mul_3/pfor/strided_slice_1/stack_1:output:0Urandom_rotation/loop_body/rotation_matrix/mul_3/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask
@random_rotation/loop_body/rotation_matrix/mul_3/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :  
;random_rotation/loop_body/rotation_matrix/mul_3/pfor/concatConcatV2Krandom_rotation/loop_body/rotation_matrix/mul_3/pfor/strided_slice:output:0Brandom_rotation/loop_body/rotation_matrix/mul_3/pfor/Tile:output:0Mrandom_rotation/loop_body/rotation_matrix/mul_3/pfor/strided_slice_1:output:0Irandom_rotation/loop_body/rotation_matrix/mul_3/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:ÿ
>random_rotation/loop_body/rotation_matrix/mul_3/pfor/Reshape_1Reshape<random_rotation/loop_body/rotation_matrix/Cos_1/pfor/Cos:y:0Drandom_rotation/loop_body/rotation_matrix/mul_3/pfor/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿï
8random_rotation/loop_body/rotation_matrix/mul_3/pfor/MulMulGrandom_rotation/loop_body/rotation_matrix/mul_3/pfor/Reshape_1:output:03random_rotation/loop_body/rotation_matrix/sub_7:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
8random_rotation/loop_body/rotation_matrix/Sin_1/pfor/SinSin9random_rotation/loop_body/stateful_uniform/pfor/AddV2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
9random_rotation/loop_body/rotation_matrix/mul_2/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :}
;random_rotation/loop_body/rotation_matrix/mul_2/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : |
:random_rotation/loop_body/rotation_matrix/mul_2/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :í
8random_rotation/loop_body/rotation_matrix/mul_2/pfor/addAddV2Drandom_rotation/loop_body/rotation_matrix/mul_2/pfor/Rank_1:output:0Crandom_rotation/loop_body/rotation_matrix/mul_2/pfor/add/y:output:0*
T0*
_output_shapes
: ê
<random_rotation/loop_body/rotation_matrix/mul_2/pfor/MaximumMaximum<random_rotation/loop_body/rotation_matrix/mul_2/pfor/add:z:0Brandom_rotation/loop_body/rotation_matrix/mul_2/pfor/Rank:output:0*
T0*
_output_shapes
: ¦
:random_rotation/loop_body/rotation_matrix/mul_2/pfor/ShapeShape<random_rotation/loop_body/rotation_matrix/Sin_1/pfor/Sin:y:0*
T0*
_output_shapes
:æ
8random_rotation/loop_body/rotation_matrix/mul_2/pfor/subSub@random_rotation/loop_body/rotation_matrix/mul_2/pfor/Maximum:z:0Brandom_rotation/loop_body/rotation_matrix/mul_2/pfor/Rank:output:0*
T0*
_output_shapes
: 
Brandom_rotation/loop_body/rotation_matrix/mul_2/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:÷
<random_rotation/loop_body/rotation_matrix/mul_2/pfor/ReshapeReshape<random_rotation/loop_body/rotation_matrix/mul_2/pfor/sub:z:0Krandom_rotation/loop_body/rotation_matrix/mul_2/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:
?random_rotation/loop_body/rotation_matrix/mul_2/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:õ
9random_rotation/loop_body/rotation_matrix/mul_2/pfor/TileTileHrandom_rotation/loop_body/rotation_matrix/mul_2/pfor/Tile/input:output:0Erandom_rotation/loop_body/rotation_matrix/mul_2/pfor/Reshape:output:0*
T0*
_output_shapes
: 
Hrandom_rotation/loop_body/rotation_matrix/mul_2/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Jrandom_rotation/loop_body/rotation_matrix/mul_2/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Jrandom_rotation/loop_body/rotation_matrix/mul_2/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ø
Brandom_rotation/loop_body/rotation_matrix/mul_2/pfor/strided_sliceStridedSliceCrandom_rotation/loop_body/rotation_matrix/mul_2/pfor/Shape:output:0Qrandom_rotation/loop_body/rotation_matrix/mul_2/pfor/strided_slice/stack:output:0Srandom_rotation/loop_body/rotation_matrix/mul_2/pfor/strided_slice/stack_1:output:0Srandom_rotation/loop_body/rotation_matrix/mul_2/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
Jrandom_rotation/loop_body/rotation_matrix/mul_2/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Lrandom_rotation/loop_body/rotation_matrix/mul_2/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Lrandom_rotation/loop_body/rotation_matrix/mul_2/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Þ
Drandom_rotation/loop_body/rotation_matrix/mul_2/pfor/strided_slice_1StridedSliceCrandom_rotation/loop_body/rotation_matrix/mul_2/pfor/Shape:output:0Srandom_rotation/loop_body/rotation_matrix/mul_2/pfor/strided_slice_1/stack:output:0Urandom_rotation/loop_body/rotation_matrix/mul_2/pfor/strided_slice_1/stack_1:output:0Urandom_rotation/loop_body/rotation_matrix/mul_2/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask
@random_rotation/loop_body/rotation_matrix/mul_2/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :  
;random_rotation/loop_body/rotation_matrix/mul_2/pfor/concatConcatV2Krandom_rotation/loop_body/rotation_matrix/mul_2/pfor/strided_slice:output:0Brandom_rotation/loop_body/rotation_matrix/mul_2/pfor/Tile:output:0Mrandom_rotation/loop_body/rotation_matrix/mul_2/pfor/strided_slice_1:output:0Irandom_rotation/loop_body/rotation_matrix/mul_2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:ÿ
>random_rotation/loop_body/rotation_matrix/mul_2/pfor/Reshape_1Reshape<random_rotation/loop_body/rotation_matrix/Sin_1/pfor/Sin:y:0Drandom_rotation/loop_body/rotation_matrix/mul_2/pfor/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿï
8random_rotation/loop_body/rotation_matrix/mul_2/pfor/MulMulGrandom_rotation/loop_body/rotation_matrix/mul_2/pfor/Reshape_1:output:03random_rotation/loop_body/rotation_matrix/sub_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
7random_rotation/loop_body/rotation_matrix/add/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :{
9random_rotation/loop_body/rotation_matrix/add/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :ì
:random_rotation/loop_body/rotation_matrix/add/pfor/MaximumMaximumBrandom_rotation/loop_body/rotation_matrix/add/pfor/Rank_1:output:0@random_rotation/loop_body/rotation_matrix/add/pfor/Rank:output:0*
T0*
_output_shapes
: ¤
8random_rotation/loop_body/rotation_matrix/add/pfor/ShapeShape<random_rotation/loop_body/rotation_matrix/mul_2/pfor/Mul:z:0*
T0*
_output_shapes
:à
6random_rotation/loop_body/rotation_matrix/add/pfor/subSub>random_rotation/loop_body/rotation_matrix/add/pfor/Maximum:z:0@random_rotation/loop_body/rotation_matrix/add/pfor/Rank:output:0*
T0*
_output_shapes
: 
@random_rotation/loop_body/rotation_matrix/add/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:ñ
:random_rotation/loop_body/rotation_matrix/add/pfor/ReshapeReshape:random_rotation/loop_body/rotation_matrix/add/pfor/sub:z:0Irandom_rotation/loop_body/rotation_matrix/add/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:
=random_rotation/loop_body/rotation_matrix/add/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:ï
7random_rotation/loop_body/rotation_matrix/add/pfor/TileTileFrandom_rotation/loop_body/rotation_matrix/add/pfor/Tile/input:output:0Crandom_rotation/loop_body/rotation_matrix/add/pfor/Reshape:output:0*
T0*
_output_shapes
: 
Frandom_rotation/loop_body/rotation_matrix/add/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Hrandom_rotation/loop_body/rotation_matrix/add/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Hrandom_rotation/loop_body/rotation_matrix/add/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Î
@random_rotation/loop_body/rotation_matrix/add/pfor/strided_sliceStridedSliceArandom_rotation/loop_body/rotation_matrix/add/pfor/Shape:output:0Orandom_rotation/loop_body/rotation_matrix/add/pfor/strided_slice/stack:output:0Qrandom_rotation/loop_body/rotation_matrix/add/pfor/strided_slice/stack_1:output:0Qrandom_rotation/loop_body/rotation_matrix/add/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
Hrandom_rotation/loop_body/rotation_matrix/add/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Jrandom_rotation/loop_body/rotation_matrix/add/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Jrandom_rotation/loop_body/rotation_matrix/add/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ô
Brandom_rotation/loop_body/rotation_matrix/add/pfor/strided_slice_1StridedSliceArandom_rotation/loop_body/rotation_matrix/add/pfor/Shape:output:0Qrandom_rotation/loop_body/rotation_matrix/add/pfor/strided_slice_1/stack:output:0Srandom_rotation/loop_body/rotation_matrix/add/pfor/strided_slice_1/stack_1:output:0Srandom_rotation/loop_body/rotation_matrix/add/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask
>random_rotation/loop_body/rotation_matrix/add/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
9random_rotation/loop_body/rotation_matrix/add/pfor/concatConcatV2Irandom_rotation/loop_body/rotation_matrix/add/pfor/strided_slice:output:0@random_rotation/loop_body/rotation_matrix/add/pfor/Tile:output:0Krandom_rotation/loop_body/rotation_matrix/add/pfor/strided_slice_1:output:0Grandom_rotation/loop_body/rotation_matrix/add/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:û
<random_rotation/loop_body/rotation_matrix/add/pfor/Reshape_1Reshape<random_rotation/loop_body/rotation_matrix/mul_2/pfor/Mul:z:0Brandom_rotation/loop_body/rotation_matrix/add/pfor/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
:random_rotation/loop_body/rotation_matrix/add/pfor/Shape_1Shape<random_rotation/loop_body/rotation_matrix/mul_3/pfor/Mul:z:0*
T0*
_output_shapes
:ä
8random_rotation/loop_body/rotation_matrix/add/pfor/sub_1Sub>random_rotation/loop_body/rotation_matrix/add/pfor/Maximum:z:0Brandom_rotation/loop_body/rotation_matrix/add/pfor/Rank_1:output:0*
T0*
_output_shapes
: 
Brandom_rotation/loop_body/rotation_matrix/add/pfor/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB:÷
<random_rotation/loop_body/rotation_matrix/add/pfor/Reshape_2Reshape<random_rotation/loop_body/rotation_matrix/add/pfor/sub_1:z:0Krandom_rotation/loop_body/rotation_matrix/add/pfor/Reshape_2/shape:output:0*
T0*
_output_shapes
:
?random_rotation/loop_body/rotation_matrix/add/pfor/Tile_1/inputConst*
_output_shapes
:*
dtype0*
valueB:õ
9random_rotation/loop_body/rotation_matrix/add/pfor/Tile_1TileHrandom_rotation/loop_body/rotation_matrix/add/pfor/Tile_1/input:output:0Erandom_rotation/loop_body/rotation_matrix/add/pfor/Reshape_2:output:0*
T0*
_output_shapes
: 
Hrandom_rotation/loop_body/rotation_matrix/add/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Jrandom_rotation/loop_body/rotation_matrix/add/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Jrandom_rotation/loop_body/rotation_matrix/add/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ø
Brandom_rotation/loop_body/rotation_matrix/add/pfor/strided_slice_2StridedSliceCrandom_rotation/loop_body/rotation_matrix/add/pfor/Shape_1:output:0Qrandom_rotation/loop_body/rotation_matrix/add/pfor/strided_slice_2/stack:output:0Srandom_rotation/loop_body/rotation_matrix/add/pfor/strided_slice_2/stack_1:output:0Srandom_rotation/loop_body/rotation_matrix/add/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
Hrandom_rotation/loop_body/rotation_matrix/add/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
Jrandom_rotation/loop_body/rotation_matrix/add/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Jrandom_rotation/loop_body/rotation_matrix/add/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ö
Brandom_rotation/loop_body/rotation_matrix/add/pfor/strided_slice_3StridedSliceCrandom_rotation/loop_body/rotation_matrix/add/pfor/Shape_1:output:0Qrandom_rotation/loop_body/rotation_matrix/add/pfor/strided_slice_3/stack:output:0Srandom_rotation/loop_body/rotation_matrix/add/pfor/strided_slice_3/stack_1:output:0Srandom_rotation/loop_body/rotation_matrix/add/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask
@random_rotation/loop_body/rotation_matrix/add/pfor/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
;random_rotation/loop_body/rotation_matrix/add/pfor/concat_1ConcatV2Krandom_rotation/loop_body/rotation_matrix/add/pfor/strided_slice_2:output:0Brandom_rotation/loop_body/rotation_matrix/add/pfor/Tile_1:output:0Krandom_rotation/loop_body/rotation_matrix/add/pfor/strided_slice_3:output:0Irandom_rotation/loop_body/rotation_matrix/add/pfor/concat_1/axis:output:0*
N*
T0*
_output_shapes
:ý
<random_rotation/loop_body/rotation_matrix/add/pfor/Reshape_3Reshape<random_rotation/loop_body/rotation_matrix/mul_3/pfor/Mul:z:0Drandom_rotation/loop_body/rotation_matrix/add/pfor/concat_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
8random_rotation/loop_body/rotation_matrix/add/pfor/AddV2AddV2Erandom_rotation/loop_body/rotation_matrix/add/pfor/Reshape_1:output:0Erandom_rotation/loop_body/rotation_matrix/add/pfor/Reshape_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
9random_rotation/loop_body/rotation_matrix/sub_8/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B : |
:random_rotation/loop_body/rotation_matrix/sub_8/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :ë
8random_rotation/loop_body/rotation_matrix/sub_8/pfor/addAddV2Brandom_rotation/loop_body/rotation_matrix/sub_8/pfor/Rank:output:0Crandom_rotation/loop_body/rotation_matrix/sub_8/pfor/add/y:output:0*
T0*
_output_shapes
: }
;random_rotation/loop_body/rotation_matrix/sub_8/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :ì
<random_rotation/loop_body/rotation_matrix/sub_8/pfor/MaximumMaximumDrandom_rotation/loop_body/rotation_matrix/sub_8/pfor/Rank_1:output:0<random_rotation/loop_body/rotation_matrix/sub_8/pfor/add:z:0*
T0*
_output_shapes
: ¦
:random_rotation/loop_body/rotation_matrix/sub_8/pfor/ShapeShape<random_rotation/loop_body/rotation_matrix/add/pfor/AddV2:z:0*
T0*
_output_shapes
:è
8random_rotation/loop_body/rotation_matrix/sub_8/pfor/subSub@random_rotation/loop_body/rotation_matrix/sub_8/pfor/Maximum:z:0Drandom_rotation/loop_body/rotation_matrix/sub_8/pfor/Rank_1:output:0*
T0*
_output_shapes
: 
Brandom_rotation/loop_body/rotation_matrix/sub_8/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:÷
<random_rotation/loop_body/rotation_matrix/sub_8/pfor/ReshapeReshape<random_rotation/loop_body/rotation_matrix/sub_8/pfor/sub:z:0Krandom_rotation/loop_body/rotation_matrix/sub_8/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:
?random_rotation/loop_body/rotation_matrix/sub_8/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:õ
9random_rotation/loop_body/rotation_matrix/sub_8/pfor/TileTileHrandom_rotation/loop_body/rotation_matrix/sub_8/pfor/Tile/input:output:0Erandom_rotation/loop_body/rotation_matrix/sub_8/pfor/Reshape:output:0*
T0*
_output_shapes
: 
Hrandom_rotation/loop_body/rotation_matrix/sub_8/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Jrandom_rotation/loop_body/rotation_matrix/sub_8/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Jrandom_rotation/loop_body/rotation_matrix/sub_8/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ø
Brandom_rotation/loop_body/rotation_matrix/sub_8/pfor/strided_sliceStridedSliceCrandom_rotation/loop_body/rotation_matrix/sub_8/pfor/Shape:output:0Qrandom_rotation/loop_body/rotation_matrix/sub_8/pfor/strided_slice/stack:output:0Srandom_rotation/loop_body/rotation_matrix/sub_8/pfor/strided_slice/stack_1:output:0Srandom_rotation/loop_body/rotation_matrix/sub_8/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
Jrandom_rotation/loop_body/rotation_matrix/sub_8/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Lrandom_rotation/loop_body/rotation_matrix/sub_8/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Lrandom_rotation/loop_body/rotation_matrix/sub_8/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Þ
Drandom_rotation/loop_body/rotation_matrix/sub_8/pfor/strided_slice_1StridedSliceCrandom_rotation/loop_body/rotation_matrix/sub_8/pfor/Shape:output:0Srandom_rotation/loop_body/rotation_matrix/sub_8/pfor/strided_slice_1/stack:output:0Urandom_rotation/loop_body/rotation_matrix/sub_8/pfor/strided_slice_1/stack_1:output:0Urandom_rotation/loop_body/rotation_matrix/sub_8/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask
@random_rotation/loop_body/rotation_matrix/sub_8/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :  
;random_rotation/loop_body/rotation_matrix/sub_8/pfor/concatConcatV2Krandom_rotation/loop_body/rotation_matrix/sub_8/pfor/strided_slice:output:0Brandom_rotation/loop_body/rotation_matrix/sub_8/pfor/Tile:output:0Mrandom_rotation/loop_body/rotation_matrix/sub_8/pfor/strided_slice_1:output:0Irandom_rotation/loop_body/rotation_matrix/sub_8/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:ÿ
>random_rotation/loop_body/rotation_matrix/sub_8/pfor/Reshape_1Reshape<random_rotation/loop_body/rotation_matrix/add/pfor/AddV2:z:0Drandom_rotation/loop_body/rotation_matrix/sub_8/pfor/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿñ
:random_rotation/loop_body/rotation_matrix/sub_8/pfor/Sub_1Sub3random_rotation/loop_body/rotation_matrix/sub_5:z:0Grandom_rotation/loop_body/rotation_matrix/sub_8/pfor/Reshape_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
=random_rotation/loop_body/rotation_matrix/truediv_1/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :
?random_rotation/loop_body/rotation_matrix/truediv_1/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : 
>random_rotation/loop_body/rotation_matrix/truediv_1/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :ù
<random_rotation/loop_body/rotation_matrix/truediv_1/pfor/addAddV2Hrandom_rotation/loop_body/rotation_matrix/truediv_1/pfor/Rank_1:output:0Grandom_rotation/loop_body/rotation_matrix/truediv_1/pfor/add/y:output:0*
T0*
_output_shapes
: ö
@random_rotation/loop_body/rotation_matrix/truediv_1/pfor/MaximumMaximum@random_rotation/loop_body/rotation_matrix/truediv_1/pfor/add:z:0Frandom_rotation/loop_body/rotation_matrix/truediv_1/pfor/Rank:output:0*
T0*
_output_shapes
: ¬
>random_rotation/loop_body/rotation_matrix/truediv_1/pfor/ShapeShape>random_rotation/loop_body/rotation_matrix/sub_8/pfor/Sub_1:z:0*
T0*
_output_shapes
:ò
<random_rotation/loop_body/rotation_matrix/truediv_1/pfor/subSubDrandom_rotation/loop_body/rotation_matrix/truediv_1/pfor/Maximum:z:0Frandom_rotation/loop_body/rotation_matrix/truediv_1/pfor/Rank:output:0*
T0*
_output_shapes
: 
Frandom_rotation/loop_body/rotation_matrix/truediv_1/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
@random_rotation/loop_body/rotation_matrix/truediv_1/pfor/ReshapeReshape@random_rotation/loop_body/rotation_matrix/truediv_1/pfor/sub:z:0Orandom_rotation/loop_body/rotation_matrix/truediv_1/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:
Crandom_rotation/loop_body/rotation_matrix/truediv_1/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:
=random_rotation/loop_body/rotation_matrix/truediv_1/pfor/TileTileLrandom_rotation/loop_body/rotation_matrix/truediv_1/pfor/Tile/input:output:0Irandom_rotation/loop_body/rotation_matrix/truediv_1/pfor/Reshape:output:0*
T0*
_output_shapes
: 
Lrandom_rotation/loop_body/rotation_matrix/truediv_1/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Nrandom_rotation/loop_body/rotation_matrix/truediv_1/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Nrandom_rotation/loop_body/rotation_matrix/truediv_1/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ì
Frandom_rotation/loop_body/rotation_matrix/truediv_1/pfor/strided_sliceStridedSliceGrandom_rotation/loop_body/rotation_matrix/truediv_1/pfor/Shape:output:0Urandom_rotation/loop_body/rotation_matrix/truediv_1/pfor/strided_slice/stack:output:0Wrandom_rotation/loop_body/rotation_matrix/truediv_1/pfor/strided_slice/stack_1:output:0Wrandom_rotation/loop_body/rotation_matrix/truediv_1/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
Nrandom_rotation/loop_body/rotation_matrix/truediv_1/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Prandom_rotation/loop_body/rotation_matrix/truediv_1/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Prandom_rotation/loop_body/rotation_matrix/truediv_1/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ò
Hrandom_rotation/loop_body/rotation_matrix/truediv_1/pfor/strided_slice_1StridedSliceGrandom_rotation/loop_body/rotation_matrix/truediv_1/pfor/Shape:output:0Wrandom_rotation/loop_body/rotation_matrix/truediv_1/pfor/strided_slice_1/stack:output:0Yrandom_rotation/loop_body/rotation_matrix/truediv_1/pfor/strided_slice_1/stack_1:output:0Yrandom_rotation/loop_body/rotation_matrix/truediv_1/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask
Drandom_rotation/loop_body/rotation_matrix/truediv_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ´
?random_rotation/loop_body/rotation_matrix/truediv_1/pfor/concatConcatV2Orandom_rotation/loop_body/rotation_matrix/truediv_1/pfor/strided_slice:output:0Frandom_rotation/loop_body/rotation_matrix/truediv_1/pfor/Tile:output:0Qrandom_rotation/loop_body/rotation_matrix/truediv_1/pfor/strided_slice_1:output:0Mrandom_rotation/loop_body/rotation_matrix/truediv_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
Brandom_rotation/loop_body/rotation_matrix/truediv_1/pfor/Reshape_1Reshape>random_rotation/loop_body/rotation_matrix/sub_8/pfor/Sub_1:z:0Hrandom_rotation/loop_body/rotation_matrix/truediv_1/pfor/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
@random_rotation/loop_body/rotation_matrix/truediv_1/pfor/RealDivRealDivKrandom_rotation/loop_body/rotation_matrix/truediv_1/pfor/Reshape_1:output:0>random_rotation/loop_body/rotation_matrix/truediv_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Nrandom_rotation/loop_body/rotation_matrix/strided_slice_6/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: 
Jrandom_rotation/loop_body/rotation_matrix/strided_slice_6/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ÷
Erandom_rotation/loop_body/rotation_matrix/strided_slice_6/pfor/concatConcatV2Wrandom_rotation/loop_body/rotation_matrix/strided_slice_6/pfor/concat/values_0:output:0Hrandom_rotation/loop_body/rotation_matrix/strided_slice_6/stack:output:0Srandom_rotation/loop_body/rotation_matrix/strided_slice_6/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
Prandom_rotation/loop_body/rotation_matrix/strided_slice_6/pfor/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB: 
Lrandom_rotation/loop_body/rotation_matrix/strided_slice_6/pfor/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ÿ
Grandom_rotation/loop_body/rotation_matrix/strided_slice_6/pfor/concat_1ConcatV2Yrandom_rotation/loop_body/rotation_matrix/strided_slice_6/pfor/concat_1/values_0:output:0Jrandom_rotation/loop_body/rotation_matrix/strided_slice_6/stack_1:output:0Urandom_rotation/loop_body/rotation_matrix/strided_slice_6/pfor/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
Prandom_rotation/loop_body/rotation_matrix/strided_slice_6/pfor/concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB:
Lrandom_rotation/loop_body/rotation_matrix/strided_slice_6/pfor/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ÿ
Grandom_rotation/loop_body/rotation_matrix/strided_slice_6/pfor/concat_2ConcatV2Yrandom_rotation/loop_body/rotation_matrix/strided_slice_6/pfor/concat_2/values_0:output:0Jrandom_rotation/loop_body/rotation_matrix/strided_slice_6/stack_2:output:0Urandom_rotation/loop_body/rotation_matrix/strided_slice_6/pfor/concat_2/axis:output:0*
N*
T0*
_output_shapes
:
Krandom_rotation/loop_body/rotation_matrix/strided_slice_6/pfor/StridedSliceStridedSliceDrandom_rotation/loop_body/rotation_matrix/truediv_1/pfor/RealDiv:z:0Nrandom_rotation/loop_body/rotation_matrix/strided_slice_6/pfor/concat:output:0Prandom_rotation/loop_body/rotation_matrix/strided_slice_6/pfor/concat_1:output:0Prandom_rotation/loop_body/rotation_matrix/strided_slice_6/pfor/concat_2:output:0*
Index0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask¬
8random_rotation/loop_body/rotation_matrix/Cos_3/pfor/CosCos9random_rotation/loop_body/stateful_uniform/pfor/AddV2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Nrandom_rotation/loop_body/rotation_matrix/strided_slice_5/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: 
Jrandom_rotation/loop_body/rotation_matrix/strided_slice_5/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ÷
Erandom_rotation/loop_body/rotation_matrix/strided_slice_5/pfor/concatConcatV2Wrandom_rotation/loop_body/rotation_matrix/strided_slice_5/pfor/concat/values_0:output:0Hrandom_rotation/loop_body/rotation_matrix/strided_slice_5/stack:output:0Srandom_rotation/loop_body/rotation_matrix/strided_slice_5/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
Prandom_rotation/loop_body/rotation_matrix/strided_slice_5/pfor/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB: 
Lrandom_rotation/loop_body/rotation_matrix/strided_slice_5/pfor/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ÿ
Grandom_rotation/loop_body/rotation_matrix/strided_slice_5/pfor/concat_1ConcatV2Yrandom_rotation/loop_body/rotation_matrix/strided_slice_5/pfor/concat_1/values_0:output:0Jrandom_rotation/loop_body/rotation_matrix/strided_slice_5/stack_1:output:0Urandom_rotation/loop_body/rotation_matrix/strided_slice_5/pfor/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
Prandom_rotation/loop_body/rotation_matrix/strided_slice_5/pfor/concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB:
Lrandom_rotation/loop_body/rotation_matrix/strided_slice_5/pfor/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ÿ
Grandom_rotation/loop_body/rotation_matrix/strided_slice_5/pfor/concat_2ConcatV2Yrandom_rotation/loop_body/rotation_matrix/strided_slice_5/pfor/concat_2/values_0:output:0Jrandom_rotation/loop_body/rotation_matrix/strided_slice_5/stack_2:output:0Urandom_rotation/loop_body/rotation_matrix/strided_slice_5/pfor/concat_2/axis:output:0*
N*
T0*
_output_shapes
:
Krandom_rotation/loop_body/rotation_matrix/strided_slice_5/pfor/StridedSliceStridedSlice<random_rotation/loop_body/rotation_matrix/Cos_3/pfor/Cos:y:0Nrandom_rotation/loop_body/rotation_matrix/strided_slice_5/pfor/concat:output:0Prandom_rotation/loop_body/rotation_matrix/strided_slice_5/pfor/concat_1:output:0Prandom_rotation/loop_body/rotation_matrix/strided_slice_5/pfor/concat_2:output:0*
Index0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask¬
8random_rotation/loop_body/rotation_matrix/Sin_3/pfor/SinSin9random_rotation/loop_body/stateful_uniform/pfor/AddV2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Nrandom_rotation/loop_body/rotation_matrix/strided_slice_4/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: 
Jrandom_rotation/loop_body/rotation_matrix/strided_slice_4/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ÷
Erandom_rotation/loop_body/rotation_matrix/strided_slice_4/pfor/concatConcatV2Wrandom_rotation/loop_body/rotation_matrix/strided_slice_4/pfor/concat/values_0:output:0Hrandom_rotation/loop_body/rotation_matrix/strided_slice_4/stack:output:0Srandom_rotation/loop_body/rotation_matrix/strided_slice_4/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
Prandom_rotation/loop_body/rotation_matrix/strided_slice_4/pfor/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB: 
Lrandom_rotation/loop_body/rotation_matrix/strided_slice_4/pfor/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ÿ
Grandom_rotation/loop_body/rotation_matrix/strided_slice_4/pfor/concat_1ConcatV2Yrandom_rotation/loop_body/rotation_matrix/strided_slice_4/pfor/concat_1/values_0:output:0Jrandom_rotation/loop_body/rotation_matrix/strided_slice_4/stack_1:output:0Urandom_rotation/loop_body/rotation_matrix/strided_slice_4/pfor/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
Prandom_rotation/loop_body/rotation_matrix/strided_slice_4/pfor/concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB:
Lrandom_rotation/loop_body/rotation_matrix/strided_slice_4/pfor/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ÿ
Grandom_rotation/loop_body/rotation_matrix/strided_slice_4/pfor/concat_2ConcatV2Yrandom_rotation/loop_body/rotation_matrix/strided_slice_4/pfor/concat_2/values_0:output:0Jrandom_rotation/loop_body/rotation_matrix/strided_slice_4/stack_2:output:0Urandom_rotation/loop_body/rotation_matrix/strided_slice_4/pfor/concat_2/axis:output:0*
N*
T0*
_output_shapes
:
Krandom_rotation/loop_body/rotation_matrix/strided_slice_4/pfor/StridedSliceStridedSlice<random_rotation/loop_body/rotation_matrix/Sin_3/pfor/Sin:y:0Nrandom_rotation/loop_body/rotation_matrix/strided_slice_4/pfor/concat:output:0Prandom_rotation/loop_body/rotation_matrix/strided_slice_4/pfor/concat_1:output:0Prandom_rotation/loop_body/rotation_matrix/strided_slice_4/pfor/concat_2:output:0*
Index0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_maskª
6random_rotation/loop_body/rotation_matrix/Sin/pfor/SinSin9random_rotation/loop_body/stateful_uniform/pfor/AddV2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
9random_rotation/loop_body/rotation_matrix/mul_1/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :}
;random_rotation/loop_body/rotation_matrix/mul_1/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : |
:random_rotation/loop_body/rotation_matrix/mul_1/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :í
8random_rotation/loop_body/rotation_matrix/mul_1/pfor/addAddV2Drandom_rotation/loop_body/rotation_matrix/mul_1/pfor/Rank_1:output:0Crandom_rotation/loop_body/rotation_matrix/mul_1/pfor/add/y:output:0*
T0*
_output_shapes
: ê
<random_rotation/loop_body/rotation_matrix/mul_1/pfor/MaximumMaximum<random_rotation/loop_body/rotation_matrix/mul_1/pfor/add:z:0Brandom_rotation/loop_body/rotation_matrix/mul_1/pfor/Rank:output:0*
T0*
_output_shapes
: ¤
:random_rotation/loop_body/rotation_matrix/mul_1/pfor/ShapeShape:random_rotation/loop_body/rotation_matrix/Sin/pfor/Sin:y:0*
T0*
_output_shapes
:æ
8random_rotation/loop_body/rotation_matrix/mul_1/pfor/subSub@random_rotation/loop_body/rotation_matrix/mul_1/pfor/Maximum:z:0Brandom_rotation/loop_body/rotation_matrix/mul_1/pfor/Rank:output:0*
T0*
_output_shapes
: 
Brandom_rotation/loop_body/rotation_matrix/mul_1/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:÷
<random_rotation/loop_body/rotation_matrix/mul_1/pfor/ReshapeReshape<random_rotation/loop_body/rotation_matrix/mul_1/pfor/sub:z:0Krandom_rotation/loop_body/rotation_matrix/mul_1/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:
?random_rotation/loop_body/rotation_matrix/mul_1/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:õ
9random_rotation/loop_body/rotation_matrix/mul_1/pfor/TileTileHrandom_rotation/loop_body/rotation_matrix/mul_1/pfor/Tile/input:output:0Erandom_rotation/loop_body/rotation_matrix/mul_1/pfor/Reshape:output:0*
T0*
_output_shapes
: 
Hrandom_rotation/loop_body/rotation_matrix/mul_1/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Jrandom_rotation/loop_body/rotation_matrix/mul_1/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Jrandom_rotation/loop_body/rotation_matrix/mul_1/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ø
Brandom_rotation/loop_body/rotation_matrix/mul_1/pfor/strided_sliceStridedSliceCrandom_rotation/loop_body/rotation_matrix/mul_1/pfor/Shape:output:0Qrandom_rotation/loop_body/rotation_matrix/mul_1/pfor/strided_slice/stack:output:0Srandom_rotation/loop_body/rotation_matrix/mul_1/pfor/strided_slice/stack_1:output:0Srandom_rotation/loop_body/rotation_matrix/mul_1/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
Jrandom_rotation/loop_body/rotation_matrix/mul_1/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Lrandom_rotation/loop_body/rotation_matrix/mul_1/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Lrandom_rotation/loop_body/rotation_matrix/mul_1/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Þ
Drandom_rotation/loop_body/rotation_matrix/mul_1/pfor/strided_slice_1StridedSliceCrandom_rotation/loop_body/rotation_matrix/mul_1/pfor/Shape:output:0Srandom_rotation/loop_body/rotation_matrix/mul_1/pfor/strided_slice_1/stack:output:0Urandom_rotation/loop_body/rotation_matrix/mul_1/pfor/strided_slice_1/stack_1:output:0Urandom_rotation/loop_body/rotation_matrix/mul_1/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask
@random_rotation/loop_body/rotation_matrix/mul_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :  
;random_rotation/loop_body/rotation_matrix/mul_1/pfor/concatConcatV2Krandom_rotation/loop_body/rotation_matrix/mul_1/pfor/strided_slice:output:0Brandom_rotation/loop_body/rotation_matrix/mul_1/pfor/Tile:output:0Mrandom_rotation/loop_body/rotation_matrix/mul_1/pfor/strided_slice_1:output:0Irandom_rotation/loop_body/rotation_matrix/mul_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:ý
>random_rotation/loop_body/rotation_matrix/mul_1/pfor/Reshape_1Reshape:random_rotation/loop_body/rotation_matrix/Sin/pfor/Sin:y:0Drandom_rotation/loop_body/rotation_matrix/mul_1/pfor/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿï
8random_rotation/loop_body/rotation_matrix/mul_1/pfor/MulMulGrandom_rotation/loop_body/rotation_matrix/mul_1/pfor/Reshape_1:output:03random_rotation/loop_body/rotation_matrix/sub_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
6random_rotation/loop_body/rotation_matrix/Cos/pfor/CosCos9random_rotation/loop_body/stateful_uniform/pfor/AddV2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
7random_rotation/loop_body/rotation_matrix/mul/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :{
9random_rotation/loop_body/rotation_matrix/mul/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : z
8random_rotation/loop_body/rotation_matrix/mul/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :ç
6random_rotation/loop_body/rotation_matrix/mul/pfor/addAddV2Brandom_rotation/loop_body/rotation_matrix/mul/pfor/Rank_1:output:0Arandom_rotation/loop_body/rotation_matrix/mul/pfor/add/y:output:0*
T0*
_output_shapes
: ä
:random_rotation/loop_body/rotation_matrix/mul/pfor/MaximumMaximum:random_rotation/loop_body/rotation_matrix/mul/pfor/add:z:0@random_rotation/loop_body/rotation_matrix/mul/pfor/Rank:output:0*
T0*
_output_shapes
: ¢
8random_rotation/loop_body/rotation_matrix/mul/pfor/ShapeShape:random_rotation/loop_body/rotation_matrix/Cos/pfor/Cos:y:0*
T0*
_output_shapes
:à
6random_rotation/loop_body/rotation_matrix/mul/pfor/subSub>random_rotation/loop_body/rotation_matrix/mul/pfor/Maximum:z:0@random_rotation/loop_body/rotation_matrix/mul/pfor/Rank:output:0*
T0*
_output_shapes
: 
@random_rotation/loop_body/rotation_matrix/mul/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:ñ
:random_rotation/loop_body/rotation_matrix/mul/pfor/ReshapeReshape:random_rotation/loop_body/rotation_matrix/mul/pfor/sub:z:0Irandom_rotation/loop_body/rotation_matrix/mul/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:
=random_rotation/loop_body/rotation_matrix/mul/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:ï
7random_rotation/loop_body/rotation_matrix/mul/pfor/TileTileFrandom_rotation/loop_body/rotation_matrix/mul/pfor/Tile/input:output:0Crandom_rotation/loop_body/rotation_matrix/mul/pfor/Reshape:output:0*
T0*
_output_shapes
: 
Frandom_rotation/loop_body/rotation_matrix/mul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Hrandom_rotation/loop_body/rotation_matrix/mul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Hrandom_rotation/loop_body/rotation_matrix/mul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Î
@random_rotation/loop_body/rotation_matrix/mul/pfor/strided_sliceStridedSliceArandom_rotation/loop_body/rotation_matrix/mul/pfor/Shape:output:0Orandom_rotation/loop_body/rotation_matrix/mul/pfor/strided_slice/stack:output:0Qrandom_rotation/loop_body/rotation_matrix/mul/pfor/strided_slice/stack_1:output:0Qrandom_rotation/loop_body/rotation_matrix/mul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
Hrandom_rotation/loop_body/rotation_matrix/mul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Jrandom_rotation/loop_body/rotation_matrix/mul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Jrandom_rotation/loop_body/rotation_matrix/mul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ô
Brandom_rotation/loop_body/rotation_matrix/mul/pfor/strided_slice_1StridedSliceArandom_rotation/loop_body/rotation_matrix/mul/pfor/Shape:output:0Qrandom_rotation/loop_body/rotation_matrix/mul/pfor/strided_slice_1/stack:output:0Srandom_rotation/loop_body/rotation_matrix/mul/pfor/strided_slice_1/stack_1:output:0Srandom_rotation/loop_body/rotation_matrix/mul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask
>random_rotation/loop_body/rotation_matrix/mul/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
9random_rotation/loop_body/rotation_matrix/mul/pfor/concatConcatV2Irandom_rotation/loop_body/rotation_matrix/mul/pfor/strided_slice:output:0@random_rotation/loop_body/rotation_matrix/mul/pfor/Tile:output:0Krandom_rotation/loop_body/rotation_matrix/mul/pfor/strided_slice_1:output:0Grandom_rotation/loop_body/rotation_matrix/mul/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:ù
<random_rotation/loop_body/rotation_matrix/mul/pfor/Reshape_1Reshape:random_rotation/loop_body/rotation_matrix/Cos/pfor/Cos:y:0Brandom_rotation/loop_body/rotation_matrix/mul/pfor/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿë
6random_rotation/loop_body/rotation_matrix/mul/pfor/MulMulErandom_rotation/loop_body/rotation_matrix/mul/pfor/Reshape_1:output:03random_rotation/loop_body/rotation_matrix/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
9random_rotation/loop_body/rotation_matrix/sub_3/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :}
;random_rotation/loop_body/rotation_matrix/sub_3/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :ò
<random_rotation/loop_body/rotation_matrix/sub_3/pfor/MaximumMaximumDrandom_rotation/loop_body/rotation_matrix/sub_3/pfor/Rank_1:output:0Brandom_rotation/loop_body/rotation_matrix/sub_3/pfor/Rank:output:0*
T0*
_output_shapes
: ¤
:random_rotation/loop_body/rotation_matrix/sub_3/pfor/ShapeShape:random_rotation/loop_body/rotation_matrix/mul/pfor/Mul:z:0*
T0*
_output_shapes
:æ
8random_rotation/loop_body/rotation_matrix/sub_3/pfor/subSub@random_rotation/loop_body/rotation_matrix/sub_3/pfor/Maximum:z:0Brandom_rotation/loop_body/rotation_matrix/sub_3/pfor/Rank:output:0*
T0*
_output_shapes
: 
Brandom_rotation/loop_body/rotation_matrix/sub_3/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:÷
<random_rotation/loop_body/rotation_matrix/sub_3/pfor/ReshapeReshape<random_rotation/loop_body/rotation_matrix/sub_3/pfor/sub:z:0Krandom_rotation/loop_body/rotation_matrix/sub_3/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:
?random_rotation/loop_body/rotation_matrix/sub_3/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:õ
9random_rotation/loop_body/rotation_matrix/sub_3/pfor/TileTileHrandom_rotation/loop_body/rotation_matrix/sub_3/pfor/Tile/input:output:0Erandom_rotation/loop_body/rotation_matrix/sub_3/pfor/Reshape:output:0*
T0*
_output_shapes
: 
Hrandom_rotation/loop_body/rotation_matrix/sub_3/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Jrandom_rotation/loop_body/rotation_matrix/sub_3/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Jrandom_rotation/loop_body/rotation_matrix/sub_3/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ø
Brandom_rotation/loop_body/rotation_matrix/sub_3/pfor/strided_sliceStridedSliceCrandom_rotation/loop_body/rotation_matrix/sub_3/pfor/Shape:output:0Qrandom_rotation/loop_body/rotation_matrix/sub_3/pfor/strided_slice/stack:output:0Srandom_rotation/loop_body/rotation_matrix/sub_3/pfor/strided_slice/stack_1:output:0Srandom_rotation/loop_body/rotation_matrix/sub_3/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
Jrandom_rotation/loop_body/rotation_matrix/sub_3/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Lrandom_rotation/loop_body/rotation_matrix/sub_3/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Lrandom_rotation/loop_body/rotation_matrix/sub_3/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Þ
Drandom_rotation/loop_body/rotation_matrix/sub_3/pfor/strided_slice_1StridedSliceCrandom_rotation/loop_body/rotation_matrix/sub_3/pfor/Shape:output:0Srandom_rotation/loop_body/rotation_matrix/sub_3/pfor/strided_slice_1/stack:output:0Urandom_rotation/loop_body/rotation_matrix/sub_3/pfor/strided_slice_1/stack_1:output:0Urandom_rotation/loop_body/rotation_matrix/sub_3/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask
@random_rotation/loop_body/rotation_matrix/sub_3/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :  
;random_rotation/loop_body/rotation_matrix/sub_3/pfor/concatConcatV2Krandom_rotation/loop_body/rotation_matrix/sub_3/pfor/strided_slice:output:0Brandom_rotation/loop_body/rotation_matrix/sub_3/pfor/Tile:output:0Mrandom_rotation/loop_body/rotation_matrix/sub_3/pfor/strided_slice_1:output:0Irandom_rotation/loop_body/rotation_matrix/sub_3/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:ý
>random_rotation/loop_body/rotation_matrix/sub_3/pfor/Reshape_1Reshape:random_rotation/loop_body/rotation_matrix/mul/pfor/Mul:z:0Drandom_rotation/loop_body/rotation_matrix/sub_3/pfor/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
<random_rotation/loop_body/rotation_matrix/sub_3/pfor/Shape_1Shape<random_rotation/loop_body/rotation_matrix/mul_1/pfor/Mul:z:0*
T0*
_output_shapes
:ê
:random_rotation/loop_body/rotation_matrix/sub_3/pfor/sub_1Sub@random_rotation/loop_body/rotation_matrix/sub_3/pfor/Maximum:z:0Drandom_rotation/loop_body/rotation_matrix/sub_3/pfor/Rank_1:output:0*
T0*
_output_shapes
: 
Drandom_rotation/loop_body/rotation_matrix/sub_3/pfor/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB:ý
>random_rotation/loop_body/rotation_matrix/sub_3/pfor/Reshape_2Reshape>random_rotation/loop_body/rotation_matrix/sub_3/pfor/sub_1:z:0Mrandom_rotation/loop_body/rotation_matrix/sub_3/pfor/Reshape_2/shape:output:0*
T0*
_output_shapes
:
Arandom_rotation/loop_body/rotation_matrix/sub_3/pfor/Tile_1/inputConst*
_output_shapes
:*
dtype0*
valueB:û
;random_rotation/loop_body/rotation_matrix/sub_3/pfor/Tile_1TileJrandom_rotation/loop_body/rotation_matrix/sub_3/pfor/Tile_1/input:output:0Grandom_rotation/loop_body/rotation_matrix/sub_3/pfor/Reshape_2:output:0*
T0*
_output_shapes
: 
Jrandom_rotation/loop_body/rotation_matrix/sub_3/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Lrandom_rotation/loop_body/rotation_matrix/sub_3/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Lrandom_rotation/loop_body/rotation_matrix/sub_3/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:â
Drandom_rotation/loop_body/rotation_matrix/sub_3/pfor/strided_slice_2StridedSliceErandom_rotation/loop_body/rotation_matrix/sub_3/pfor/Shape_1:output:0Srandom_rotation/loop_body/rotation_matrix/sub_3/pfor/strided_slice_2/stack:output:0Urandom_rotation/loop_body/rotation_matrix/sub_3/pfor/strided_slice_2/stack_1:output:0Urandom_rotation/loop_body/rotation_matrix/sub_3/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
Jrandom_rotation/loop_body/rotation_matrix/sub_3/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
Lrandom_rotation/loop_body/rotation_matrix/sub_3/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Lrandom_rotation/loop_body/rotation_matrix/sub_3/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:à
Drandom_rotation/loop_body/rotation_matrix/sub_3/pfor/strided_slice_3StridedSliceErandom_rotation/loop_body/rotation_matrix/sub_3/pfor/Shape_1:output:0Srandom_rotation/loop_body/rotation_matrix/sub_3/pfor/strided_slice_3/stack:output:0Urandom_rotation/loop_body/rotation_matrix/sub_3/pfor/strided_slice_3/stack_1:output:0Urandom_rotation/loop_body/rotation_matrix/sub_3/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask
Brandom_rotation/loop_body/rotation_matrix/sub_3/pfor/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¨
=random_rotation/loop_body/rotation_matrix/sub_3/pfor/concat_1ConcatV2Mrandom_rotation/loop_body/rotation_matrix/sub_3/pfor/strided_slice_2:output:0Drandom_rotation/loop_body/rotation_matrix/sub_3/pfor/Tile_1:output:0Mrandom_rotation/loop_body/rotation_matrix/sub_3/pfor/strided_slice_3:output:0Krandom_rotation/loop_body/rotation_matrix/sub_3/pfor/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
>random_rotation/loop_body/rotation_matrix/sub_3/pfor/Reshape_3Reshape<random_rotation/loop_body/rotation_matrix/mul_1/pfor/Mul:z:0Frandom_rotation/loop_body/rotation_matrix/sub_3/pfor/concat_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:random_rotation/loop_body/rotation_matrix/sub_3/pfor/Sub_2SubGrandom_rotation/loop_body/rotation_matrix/sub_3/pfor/Reshape_1:output:0Grandom_rotation/loop_body/rotation_matrix/sub_3/pfor/Reshape_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
9random_rotation/loop_body/rotation_matrix/sub_4/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B : |
:random_rotation/loop_body/rotation_matrix/sub_4/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :ë
8random_rotation/loop_body/rotation_matrix/sub_4/pfor/addAddV2Brandom_rotation/loop_body/rotation_matrix/sub_4/pfor/Rank:output:0Crandom_rotation/loop_body/rotation_matrix/sub_4/pfor/add/y:output:0*
T0*
_output_shapes
: }
;random_rotation/loop_body/rotation_matrix/sub_4/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :ì
<random_rotation/loop_body/rotation_matrix/sub_4/pfor/MaximumMaximumDrandom_rotation/loop_body/rotation_matrix/sub_4/pfor/Rank_1:output:0<random_rotation/loop_body/rotation_matrix/sub_4/pfor/add:z:0*
T0*
_output_shapes
: ¨
:random_rotation/loop_body/rotation_matrix/sub_4/pfor/ShapeShape>random_rotation/loop_body/rotation_matrix/sub_3/pfor/Sub_2:z:0*
T0*
_output_shapes
:è
8random_rotation/loop_body/rotation_matrix/sub_4/pfor/subSub@random_rotation/loop_body/rotation_matrix/sub_4/pfor/Maximum:z:0Drandom_rotation/loop_body/rotation_matrix/sub_4/pfor/Rank_1:output:0*
T0*
_output_shapes
: 
Brandom_rotation/loop_body/rotation_matrix/sub_4/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:÷
<random_rotation/loop_body/rotation_matrix/sub_4/pfor/ReshapeReshape<random_rotation/loop_body/rotation_matrix/sub_4/pfor/sub:z:0Krandom_rotation/loop_body/rotation_matrix/sub_4/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:
?random_rotation/loop_body/rotation_matrix/sub_4/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:õ
9random_rotation/loop_body/rotation_matrix/sub_4/pfor/TileTileHrandom_rotation/loop_body/rotation_matrix/sub_4/pfor/Tile/input:output:0Erandom_rotation/loop_body/rotation_matrix/sub_4/pfor/Reshape:output:0*
T0*
_output_shapes
: 
Hrandom_rotation/loop_body/rotation_matrix/sub_4/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Jrandom_rotation/loop_body/rotation_matrix/sub_4/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Jrandom_rotation/loop_body/rotation_matrix/sub_4/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ø
Brandom_rotation/loop_body/rotation_matrix/sub_4/pfor/strided_sliceStridedSliceCrandom_rotation/loop_body/rotation_matrix/sub_4/pfor/Shape:output:0Qrandom_rotation/loop_body/rotation_matrix/sub_4/pfor/strided_slice/stack:output:0Srandom_rotation/loop_body/rotation_matrix/sub_4/pfor/strided_slice/stack_1:output:0Srandom_rotation/loop_body/rotation_matrix/sub_4/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
Jrandom_rotation/loop_body/rotation_matrix/sub_4/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Lrandom_rotation/loop_body/rotation_matrix/sub_4/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Lrandom_rotation/loop_body/rotation_matrix/sub_4/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Þ
Drandom_rotation/loop_body/rotation_matrix/sub_4/pfor/strided_slice_1StridedSliceCrandom_rotation/loop_body/rotation_matrix/sub_4/pfor/Shape:output:0Srandom_rotation/loop_body/rotation_matrix/sub_4/pfor/strided_slice_1/stack:output:0Urandom_rotation/loop_body/rotation_matrix/sub_4/pfor/strided_slice_1/stack_1:output:0Urandom_rotation/loop_body/rotation_matrix/sub_4/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask
@random_rotation/loop_body/rotation_matrix/sub_4/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :  
;random_rotation/loop_body/rotation_matrix/sub_4/pfor/concatConcatV2Krandom_rotation/loop_body/rotation_matrix/sub_4/pfor/strided_slice:output:0Brandom_rotation/loop_body/rotation_matrix/sub_4/pfor/Tile:output:0Mrandom_rotation/loop_body/rotation_matrix/sub_4/pfor/strided_slice_1:output:0Irandom_rotation/loop_body/rotation_matrix/sub_4/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
>random_rotation/loop_body/rotation_matrix/sub_4/pfor/Reshape_1Reshape>random_rotation/loop_body/rotation_matrix/sub_3/pfor/Sub_2:z:0Drandom_rotation/loop_body/rotation_matrix/sub_4/pfor/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿï
:random_rotation/loop_body/rotation_matrix/sub_4/pfor/Sub_1Sub1random_rotation/loop_body/rotation_matrix/sub:z:0Grandom_rotation/loop_body/rotation_matrix/sub_4/pfor/Reshape_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
;random_rotation/loop_body/rotation_matrix/truediv/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :
=random_rotation/loop_body/rotation_matrix/truediv/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : ~
<random_rotation/loop_body/rotation_matrix/truediv/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :ó
:random_rotation/loop_body/rotation_matrix/truediv/pfor/addAddV2Frandom_rotation/loop_body/rotation_matrix/truediv/pfor/Rank_1:output:0Erandom_rotation/loop_body/rotation_matrix/truediv/pfor/add/y:output:0*
T0*
_output_shapes
: ð
>random_rotation/loop_body/rotation_matrix/truediv/pfor/MaximumMaximum>random_rotation/loop_body/rotation_matrix/truediv/pfor/add:z:0Drandom_rotation/loop_body/rotation_matrix/truediv/pfor/Rank:output:0*
T0*
_output_shapes
: ª
<random_rotation/loop_body/rotation_matrix/truediv/pfor/ShapeShape>random_rotation/loop_body/rotation_matrix/sub_4/pfor/Sub_1:z:0*
T0*
_output_shapes
:ì
:random_rotation/loop_body/rotation_matrix/truediv/pfor/subSubBrandom_rotation/loop_body/rotation_matrix/truediv/pfor/Maximum:z:0Drandom_rotation/loop_body/rotation_matrix/truediv/pfor/Rank:output:0*
T0*
_output_shapes
: 
Drandom_rotation/loop_body/rotation_matrix/truediv/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:ý
>random_rotation/loop_body/rotation_matrix/truediv/pfor/ReshapeReshape>random_rotation/loop_body/rotation_matrix/truediv/pfor/sub:z:0Mrandom_rotation/loop_body/rotation_matrix/truediv/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:
Arandom_rotation/loop_body/rotation_matrix/truediv/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:û
;random_rotation/loop_body/rotation_matrix/truediv/pfor/TileTileJrandom_rotation/loop_body/rotation_matrix/truediv/pfor/Tile/input:output:0Grandom_rotation/loop_body/rotation_matrix/truediv/pfor/Reshape:output:0*
T0*
_output_shapes
: 
Jrandom_rotation/loop_body/rotation_matrix/truediv/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Lrandom_rotation/loop_body/rotation_matrix/truediv/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Lrandom_rotation/loop_body/rotation_matrix/truediv/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:â
Drandom_rotation/loop_body/rotation_matrix/truediv/pfor/strided_sliceStridedSliceErandom_rotation/loop_body/rotation_matrix/truediv/pfor/Shape:output:0Srandom_rotation/loop_body/rotation_matrix/truediv/pfor/strided_slice/stack:output:0Urandom_rotation/loop_body/rotation_matrix/truediv/pfor/strided_slice/stack_1:output:0Urandom_rotation/loop_body/rotation_matrix/truediv/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
Lrandom_rotation/loop_body/rotation_matrix/truediv/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
Nrandom_rotation/loop_body/rotation_matrix/truediv/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
Nrandom_rotation/loop_body/rotation_matrix/truediv/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:è
Frandom_rotation/loop_body/rotation_matrix/truediv/pfor/strided_slice_1StridedSliceErandom_rotation/loop_body/rotation_matrix/truediv/pfor/Shape:output:0Urandom_rotation/loop_body/rotation_matrix/truediv/pfor/strided_slice_1/stack:output:0Wrandom_rotation/loop_body/rotation_matrix/truediv/pfor/strided_slice_1/stack_1:output:0Wrandom_rotation/loop_body/rotation_matrix/truediv/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask
Brandom_rotation/loop_body/rotation_matrix/truediv/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ª
=random_rotation/loop_body/rotation_matrix/truediv/pfor/concatConcatV2Mrandom_rotation/loop_body/rotation_matrix/truediv/pfor/strided_slice:output:0Drandom_rotation/loop_body/rotation_matrix/truediv/pfor/Tile:output:0Orandom_rotation/loop_body/rotation_matrix/truediv/pfor/strided_slice_1:output:0Krandom_rotation/loop_body/rotation_matrix/truediv/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
@random_rotation/loop_body/rotation_matrix/truediv/pfor/Reshape_1Reshape>random_rotation/loop_body/rotation_matrix/sub_4/pfor/Sub_1:z:0Frandom_rotation/loop_body/rotation_matrix/truediv/pfor/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
>random_rotation/loop_body/rotation_matrix/truediv/pfor/RealDivRealDivIrandom_rotation/loop_body/rotation_matrix/truediv/pfor/Reshape_1:output:0<random_rotation/loop_body/rotation_matrix/truediv/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Nrandom_rotation/loop_body/rotation_matrix/strided_slice_3/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: 
Jrandom_rotation/loop_body/rotation_matrix/strided_slice_3/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ÷
Erandom_rotation/loop_body/rotation_matrix/strided_slice_3/pfor/concatConcatV2Wrandom_rotation/loop_body/rotation_matrix/strided_slice_3/pfor/concat/values_0:output:0Hrandom_rotation/loop_body/rotation_matrix/strided_slice_3/stack:output:0Srandom_rotation/loop_body/rotation_matrix/strided_slice_3/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
Prandom_rotation/loop_body/rotation_matrix/strided_slice_3/pfor/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB: 
Lrandom_rotation/loop_body/rotation_matrix/strided_slice_3/pfor/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ÿ
Grandom_rotation/loop_body/rotation_matrix/strided_slice_3/pfor/concat_1ConcatV2Yrandom_rotation/loop_body/rotation_matrix/strided_slice_3/pfor/concat_1/values_0:output:0Jrandom_rotation/loop_body/rotation_matrix/strided_slice_3/stack_1:output:0Urandom_rotation/loop_body/rotation_matrix/strided_slice_3/pfor/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
Prandom_rotation/loop_body/rotation_matrix/strided_slice_3/pfor/concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB:
Lrandom_rotation/loop_body/rotation_matrix/strided_slice_3/pfor/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ÿ
Grandom_rotation/loop_body/rotation_matrix/strided_slice_3/pfor/concat_2ConcatV2Yrandom_rotation/loop_body/rotation_matrix/strided_slice_3/pfor/concat_2/values_0:output:0Jrandom_rotation/loop_body/rotation_matrix/strided_slice_3/stack_2:output:0Urandom_rotation/loop_body/rotation_matrix/strided_slice_3/pfor/concat_2/axis:output:0*
N*
T0*
_output_shapes
:
Krandom_rotation/loop_body/rotation_matrix/strided_slice_3/pfor/StridedSliceStridedSliceBrandom_rotation/loop_body/rotation_matrix/truediv/pfor/RealDiv:z:0Nrandom_rotation/loop_body/rotation_matrix/strided_slice_3/pfor/concat:output:0Prandom_rotation/loop_body/rotation_matrix/strided_slice_3/pfor/concat_1:output:0Prandom_rotation/loop_body/rotation_matrix/strided_slice_3/pfor/concat_2:output:0*
Index0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask¬
8random_rotation/loop_body/rotation_matrix/Sin_2/pfor/SinSin9random_rotation/loop_body/stateful_uniform/pfor/AddV2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Nrandom_rotation/loop_body/rotation_matrix/strided_slice_2/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: 
Jrandom_rotation/loop_body/rotation_matrix/strided_slice_2/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ÷
Erandom_rotation/loop_body/rotation_matrix/strided_slice_2/pfor/concatConcatV2Wrandom_rotation/loop_body/rotation_matrix/strided_slice_2/pfor/concat/values_0:output:0Hrandom_rotation/loop_body/rotation_matrix/strided_slice_2/stack:output:0Srandom_rotation/loop_body/rotation_matrix/strided_slice_2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
Prandom_rotation/loop_body/rotation_matrix/strided_slice_2/pfor/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB: 
Lrandom_rotation/loop_body/rotation_matrix/strided_slice_2/pfor/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ÿ
Grandom_rotation/loop_body/rotation_matrix/strided_slice_2/pfor/concat_1ConcatV2Yrandom_rotation/loop_body/rotation_matrix/strided_slice_2/pfor/concat_1/values_0:output:0Jrandom_rotation/loop_body/rotation_matrix/strided_slice_2/stack_1:output:0Urandom_rotation/loop_body/rotation_matrix/strided_slice_2/pfor/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
Prandom_rotation/loop_body/rotation_matrix/strided_slice_2/pfor/concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB:
Lrandom_rotation/loop_body/rotation_matrix/strided_slice_2/pfor/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ÿ
Grandom_rotation/loop_body/rotation_matrix/strided_slice_2/pfor/concat_2ConcatV2Yrandom_rotation/loop_body/rotation_matrix/strided_slice_2/pfor/concat_2/values_0:output:0Jrandom_rotation/loop_body/rotation_matrix/strided_slice_2/stack_2:output:0Urandom_rotation/loop_body/rotation_matrix/strided_slice_2/pfor/concat_2/axis:output:0*
N*
T0*
_output_shapes
:
Krandom_rotation/loop_body/rotation_matrix/strided_slice_2/pfor/StridedSliceStridedSlice<random_rotation/loop_body/rotation_matrix/Sin_2/pfor/Sin:y:0Nrandom_rotation/loop_body/rotation_matrix/strided_slice_2/pfor/concat:output:0Prandom_rotation/loop_body/rotation_matrix/strided_slice_2/pfor/concat_1:output:0Prandom_rotation/loop_body/rotation_matrix/strided_slice_2/pfor/concat_2:output:0*
Index0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_maskÉ
6random_rotation/loop_body/rotation_matrix/Neg/pfor/NegNegTrandom_rotation/loop_body/rotation_matrix/strided_slice_2/pfor/StridedSlice:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
8random_rotation/loop_body/rotation_matrix/Cos_2/pfor/CosCos9random_rotation/loop_body/stateful_uniform/pfor/AddV2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Nrandom_rotation/loop_body/rotation_matrix/strided_slice_1/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: 
Jrandom_rotation/loop_body/rotation_matrix/strided_slice_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ÷
Erandom_rotation/loop_body/rotation_matrix/strided_slice_1/pfor/concatConcatV2Wrandom_rotation/loop_body/rotation_matrix/strided_slice_1/pfor/concat/values_0:output:0Hrandom_rotation/loop_body/rotation_matrix/strided_slice_1/stack:output:0Srandom_rotation/loop_body/rotation_matrix/strided_slice_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
Prandom_rotation/loop_body/rotation_matrix/strided_slice_1/pfor/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB: 
Lrandom_rotation/loop_body/rotation_matrix/strided_slice_1/pfor/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ÿ
Grandom_rotation/loop_body/rotation_matrix/strided_slice_1/pfor/concat_1ConcatV2Yrandom_rotation/loop_body/rotation_matrix/strided_slice_1/pfor/concat_1/values_0:output:0Jrandom_rotation/loop_body/rotation_matrix/strided_slice_1/stack_1:output:0Urandom_rotation/loop_body/rotation_matrix/strided_slice_1/pfor/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
Prandom_rotation/loop_body/rotation_matrix/strided_slice_1/pfor/concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB:
Lrandom_rotation/loop_body/rotation_matrix/strided_slice_1/pfor/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ÿ
Grandom_rotation/loop_body/rotation_matrix/strided_slice_1/pfor/concat_2ConcatV2Yrandom_rotation/loop_body/rotation_matrix/strided_slice_1/pfor/concat_2/values_0:output:0Jrandom_rotation/loop_body/rotation_matrix/strided_slice_1/stack_2:output:0Urandom_rotation/loop_body/rotation_matrix/strided_slice_1/pfor/concat_2/axis:output:0*
N*
T0*
_output_shapes
:
Krandom_rotation/loop_body/rotation_matrix/strided_slice_1/pfor/StridedSliceStridedSlice<random_rotation/loop_body/rotation_matrix/Cos_2/pfor/Cos:y:0Nrandom_rotation/loop_body/rotation_matrix/strided_slice_1/pfor/concat:output:0Prandom_rotation/loop_body/rotation_matrix/strided_slice_1/pfor/concat_1:output:0Prandom_rotation/loop_body/rotation_matrix/strided_slice_1/pfor/concat_2:output:0*
Index0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask
;random_rotation/loop_body/rotation_matrix/concat/pfor/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      
Urandom_rotation/loop_body/rotation_matrix/concat/pfor/ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:
Erandom_rotation/loop_body/rotation_matrix/concat/pfor/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :
?random_rotation/loop_body/rotation_matrix/concat/pfor/ones_likeFill^random_rotation/loop_body/rotation_matrix/concat/pfor/ones_like/Shape/shape_as_tensor:output:0Nrandom_rotation/loop_body/rotation_matrix/concat/pfor/ones_like/Const:output:0*
T0*
_output_shapes
:
Crandom_rotation/loop_body/rotation_matrix/concat/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ
=random_rotation/loop_body/rotation_matrix/concat/pfor/ReshapeReshapeHrandom_rotation/loop_body/rotation_matrix/concat/pfor/ones_like:output:0Lrandom_rotation/loop_body/rotation_matrix/concat/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:
Erandom_rotation/loop_body/rotation_matrix/concat/pfor/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿæ
?random_rotation/loop_body/rotation_matrix/concat/pfor/Reshape_1Reshape%random_rotation/pfor/Reshape:output:0Nrandom_rotation/loop_body/rotation_matrix/concat/pfor/Reshape_1/shape:output:0*
T0*
_output_shapes
:
Arandom_rotation/loop_body/rotation_matrix/concat/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ô
<random_rotation/loop_body/rotation_matrix/concat/pfor/concatConcatV2Hrandom_rotation/loop_body/rotation_matrix/concat/pfor/Reshape_1:output:0Frandom_rotation/loop_body/rotation_matrix/concat/pfor/Reshape:output:0Jrandom_rotation/loop_body/rotation_matrix/concat/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
Drandom_rotation/loop_body/rotation_matrix/concat/pfor/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
@random_rotation/loop_body/rotation_matrix/concat/pfor/ExpandDims
ExpandDims8random_rotation/loop_body/rotation_matrix/zeros:output:0Mrandom_rotation/loop_body/rotation_matrix/concat/pfor/ExpandDims/dim:output:0*
T0*"
_output_shapes
:
:random_rotation/loop_body/rotation_matrix/concat/pfor/TileTileIrandom_rotation/loop_body/rotation_matrix/concat/pfor/ExpandDims:output:0Erandom_rotation/loop_body/rotation_matrix/concat/pfor/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Drandom_rotation/loop_body/rotation_matrix/concat/pfor/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : 
Brandom_rotation/loop_body/rotation_matrix/concat/pfor/GreaterEqualGreaterEqual>random_rotation/loop_body/rotation_matrix/concat/axis:output:0Mrandom_rotation/loop_body/rotation_matrix/concat/pfor/GreaterEqual/y:output:0*
T0*
_output_shapes
: º
:random_rotation/loop_body/rotation_matrix/concat/pfor/CastCastFrandom_rotation/loop_body/rotation_matrix/concat/pfor/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
: ã
9random_rotation/loop_body/rotation_matrix/concat/pfor/addAddV2>random_rotation/loop_body/rotation_matrix/concat/axis:output:0>random_rotation/loop_body/rotation_matrix/concat/pfor/Cast:y:0*
T0*
_output_shapes
: ÷
>random_rotation/loop_body/rotation_matrix/concat/pfor/concat_1ConcatV2Trandom_rotation/loop_body/rotation_matrix/strided_slice_1/pfor/StridedSlice:output:0:random_rotation/loop_body/rotation_matrix/Neg/pfor/Neg:y:0Trandom_rotation/loop_body/rotation_matrix/strided_slice_3/pfor/StridedSlice:output:0Trandom_rotation/loop_body/rotation_matrix/strided_slice_4/pfor/StridedSlice:output:0Trandom_rotation/loop_body/rotation_matrix/strided_slice_5/pfor/StridedSlice:output:0Trandom_rotation/loop_body/rotation_matrix/strided_slice_6/pfor/StridedSlice:output:0Crandom_rotation/loop_body/rotation_matrix/concat/pfor/Tile:output:0=random_rotation/loop_body/rotation_matrix/concat/pfor/add:z:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
,random_rotation/loop_body/SelectV2/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B : o
-random_rotation/loop_body/SelectV2/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ä
+random_rotation/loop_body/SelectV2/pfor/addAddV25random_rotation/loop_body/SelectV2/pfor/Rank:output:06random_rotation/loop_body/SelectV2/pfor/add/y:output:0*
T0*
_output_shapes
: p
.random_rotation/loop_body/SelectV2/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :p
.random_rotation/loop_body/SelectV2/pfor/Rank_2Const*
_output_shapes
: *
dtype0*
value	B : q
/random_rotation/loop_body/SelectV2/pfor/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Ê
-random_rotation/loop_body/SelectV2/pfor/add_1AddV27random_rotation/loop_body/SelectV2/pfor/Rank_2:output:08random_rotation/loop_body/SelectV2/pfor/add_1/y:output:0*
T0*
_output_shapes
: Å
/random_rotation/loop_body/SelectV2/pfor/MaximumMaximum7random_rotation/loop_body/SelectV2/pfor/Rank_1:output:0/random_rotation/loop_body/SelectV2/pfor/add:z:0*
T0*
_output_shapes
: Å
1random_rotation/loop_body/SelectV2/pfor/Maximum_1Maximum1random_rotation/loop_body/SelectV2/pfor/add_1:z:03random_rotation/loop_body/SelectV2/pfor/Maximum:z:0*
T0*
_output_shapes
: 
-random_rotation/loop_body/SelectV2/pfor/ShapeShape#random_rotation/pfor/range:output:0*
T0*
_output_shapes
:Ã
+random_rotation/loop_body/SelectV2/pfor/subSub5random_rotation/loop_body/SelectV2/pfor/Maximum_1:z:07random_rotation/loop_body/SelectV2/pfor/Rank_1:output:0*
T0*
_output_shapes
: 
5random_rotation/loop_body/SelectV2/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:Ð
/random_rotation/loop_body/SelectV2/pfor/ReshapeReshape/random_rotation/loop_body/SelectV2/pfor/sub:z:0>random_rotation/loop_body/SelectV2/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:|
2random_rotation/loop_body/SelectV2/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:Î
,random_rotation/loop_body/SelectV2/pfor/TileTile;random_rotation/loop_body/SelectV2/pfor/Tile/input:output:08random_rotation/loop_body/SelectV2/pfor/Reshape:output:0*
T0*
_output_shapes
: 
;random_rotation/loop_body/SelectV2/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=random_rotation/loop_body/SelectV2/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
=random_rotation/loop_body/SelectV2/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
5random_rotation/loop_body/SelectV2/pfor/strided_sliceStridedSlice6random_rotation/loop_body/SelectV2/pfor/Shape:output:0Drandom_rotation/loop_body/SelectV2/pfor/strided_slice/stack:output:0Frandom_rotation/loop_body/SelectV2/pfor/strided_slice/stack_1:output:0Frandom_rotation/loop_body/SelectV2/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
=random_rotation/loop_body/SelectV2/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
?random_rotation/loop_body/SelectV2/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
?random_rotation/loop_body/SelectV2/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
7random_rotation/loop_body/SelectV2/pfor/strided_slice_1StridedSlice6random_rotation/loop_body/SelectV2/pfor/Shape:output:0Frandom_rotation/loop_body/SelectV2/pfor/strided_slice_1/stack:output:0Hrandom_rotation/loop_body/SelectV2/pfor/strided_slice_1/stack_1:output:0Hrandom_rotation/loop_body/SelectV2/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_masku
3random_rotation/loop_body/SelectV2/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
.random_rotation/loop_body/SelectV2/pfor/concatConcatV2>random_rotation/loop_body/SelectV2/pfor/strided_slice:output:05random_rotation/loop_body/SelectV2/pfor/Tile:output:0@random_rotation/loop_body/SelectV2/pfor/strided_slice_1:output:0<random_rotation/loop_body/SelectV2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:È
1random_rotation/loop_body/SelectV2/pfor/Reshape_1Reshape#random_rotation/pfor/range:output:07random_rotation/loop_body/SelectV2/pfor/concat:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿü
0random_rotation/loop_body/SelectV2/pfor/SelectV2SelectV2%random_rotation/loop_body/Greater:z:0:random_rotation/loop_body/SelectV2/pfor/Reshape_1:output:0-random_rotation/loop_body/SelectV2/e:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
5random_rotation/loop_body/GatherV2/pfor/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ó
0random_rotation/loop_body/GatherV2/pfor/GatherV2GatherV2;random_flip/map/TensorArrayV2Stack/TensorListStack:tensor:09random_rotation/loop_body/SelectV2/pfor/SelectV2:output:0>random_rotation/loop_body/GatherV2/pfor/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
8random_rotation/loop_body/ExpandDims/pfor/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : Ý
6random_rotation/loop_body/ExpandDims/pfor/GreaterEqualGreaterEqual1random_rotation/loop_body/ExpandDims/dim:output:0Arandom_rotation/loop_body/ExpandDims/pfor/GreaterEqual/y:output:0*
T0*
_output_shapes
: ¢
.random_rotation/loop_body/ExpandDims/pfor/CastCast:random_rotation/loop_body/ExpandDims/pfor/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
: ¾
-random_rotation/loop_body/ExpandDims/pfor/addAddV21random_rotation/loop_body/ExpandDims/dim:output:02random_rotation/loop_body/ExpandDims/pfor/Cast:y:0*
T0*
_output_shapes
: ð
4random_rotation/loop_body/ExpandDims/pfor/ExpandDims
ExpandDims9random_rotation/loop_body/GatherV2/pfor/GatherV2:output:01random_rotation/loop_body/ExpandDims/pfor/add:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿ¡
Wrandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: £
Yrandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:£
Yrandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ø
Qrandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/strided_sliceStridedSlice%random_rotation/pfor/Reshape:output:0`random_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/strided_slice/stack:output:0brandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/strided_slice/stack_1:output:0brandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskª
_random_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿþ
Qrandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/TensorArrayV2TensorListReservehrandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/TensorArrayV2/element_shape:output:0Zrandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Irandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : §
\random_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Vrandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¾

Irandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/whileStatelessWhile_random_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/loop_counter:output:0erandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/maximum_iterations:output:0Rrandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/Const:output:0Zrandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/TensorArrayV2:handle:0Zrandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/strided_slice:output:0=random_rotation/loop_body/ExpandDims/pfor/ExpandDims:output:0Grandom_rotation/loop_body/rotation_matrix/concat/pfor/concat_1:output:0:random_rotation/loop_body/transform/strided_slice:output:07random_rotation/loop_body/transform/fill_value:output:0*
T
2	*
_lower_using_switch_merge(*
_num_original_outputs	*^
_output_shapesL
J: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: * 
_read_only_resource_inputs
 *
_stateful_parallelism( *_
bodyWRU
Srandom_rotation_loop_body_transform_ImageProjectiveTransformV3_pfor_while_body_7679*_
condWRU
Srandom_rotation_loop_body_transform_ImageProjectiveTransformV3_pfor_while_cond_7678*]
output_shapesL
J: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: 
Krandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 Á
drandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*)
value B"ÿÿÿÿ   ÿÿÿÿÿÿÿÿ   ò
Vrandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/TensorListConcatV2TensorListConcatV2Rrandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while:output:3mrandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/TensorListConcatV2/element_shape:output:0Trandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/Const_1:output:0*D
_output_shapes2
0:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0í
.random_rotation/loop_body/Squeeze/pfor/SqueezeSqueeze_random_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/TensorListConcatV2:tensor:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims
U
rescaling/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *<]
rescaling/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿe
rescaling/Cast_1Castrescaling/Cast_1/x:output:0*

DstT0*

SrcT0*
_output_shapes
: ¤
rescaling/mulMul7random_rotation/loop_body/Squeeze/pfor/Squeeze:output:0rescaling/Cast/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
rescaling/addAddV2rescaling/mul:z:0rescaling/Cast_1:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0µ
conv2d/Conv2DConv2Drescaling/add:z:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿüü*
paddingVALID*
strides

conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿüüh
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿüü¨
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~~*
ksize
*
paddingVALID*
strides

conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype0Ä
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿzz0*
paddingVALID*
strides

conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿzz0j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿzz0¬
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ==0*
ksize
*
paddingVALID*
strides

conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:0`*
dtype0Æ
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ99`*
paddingVALID*
strides

conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ99`j
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ99`^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`Â 
flatten/ReshapeReshapeconv2d_2/Relu:activations:0flatten/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿàZ
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   A
dropout/dropout/MulMulflatten/Reshape:output:0dropout/dropout/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿà]
dropout/dropout/ShapeShapeflatten/Reshape:output:0*
T0*
_output_shapes
:
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*
dtype0c
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?À
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
à*
dtype0
dense/MatMulMatMuldropout/dropout/Mul_1:z:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
IdentityIdentitydense/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿË
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp:^random_rotation/loop_body/stateful_uniform/RngReadAndSkipE^random_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2v
9random_rotation/loop_body/stateful_uniform/RngReadAndSkip9random_rotation/loop_body/stateful_uniform/RngReadAndSkip2
Drandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/whileDrandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/while:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ç4
	
9loop_body_stateful_uniform_Bitcast_1_pfor_while_body_8167p
lloop_body_stateful_uniform_bitcast_1_pfor_while_loop_body_stateful_uniform_bitcast_1_pfor_while_loop_counterv
rloop_body_stateful_uniform_bitcast_1_pfor_while_loop_body_stateful_uniform_bitcast_1_pfor_while_maximum_iterations?
;loop_body_stateful_uniform_bitcast_1_pfor_while_placeholderA
=loop_body_stateful_uniform_bitcast_1_pfor_while_placeholder_1m
iloop_body_stateful_uniform_bitcast_1_pfor_while_loop_body_stateful_uniform_bitcast_1_pfor_strided_slice_0
|loop_body_stateful_uniform_bitcast_1_pfor_while_strided_slice_loop_body_stateful_uniform_strided_slice_1_pfor_stridedslice_0	<
8loop_body_stateful_uniform_bitcast_1_pfor_while_identity>
:loop_body_stateful_uniform_bitcast_1_pfor_while_identity_1>
:loop_body_stateful_uniform_bitcast_1_pfor_while_identity_2>
:loop_body_stateful_uniform_bitcast_1_pfor_while_identity_3k
gloop_body_stateful_uniform_bitcast_1_pfor_while_loop_body_stateful_uniform_bitcast_1_pfor_strided_slice~
zloop_body_stateful_uniform_bitcast_1_pfor_while_strided_slice_loop_body_stateful_uniform_strided_slice_1_pfor_stridedslice	w
5loop_body/stateful_uniform/Bitcast_1/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ú
3loop_body/stateful_uniform/Bitcast_1/pfor/while/addAddV2;loop_body_stateful_uniform_bitcast_1_pfor_while_placeholder>loop_body/stateful_uniform/Bitcast_1/pfor/while/add/y:output:0*
T0*
_output_shapes
: 
Eloop_body/stateful_uniform/Bitcast_1/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 
Cloop_body/stateful_uniform/Bitcast_1/pfor/while/strided_slice/stackPack;loop_body_stateful_uniform_bitcast_1_pfor_while_placeholderNloop_body/stateful_uniform/Bitcast_1/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
:
Gloop_body/stateful_uniform/Bitcast_1/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : 
Eloop_body/stateful_uniform/Bitcast_1/pfor/while/strided_slice/stack_1Pack7loop_body/stateful_uniform/Bitcast_1/pfor/while/add:z:0Ploop_body/stateful_uniform/Bitcast_1/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:
Eloop_body/stateful_uniform/Bitcast_1/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
=loop_body/stateful_uniform/Bitcast_1/pfor/while/strided_sliceStridedSlice|loop_body_stateful_uniform_bitcast_1_pfor_while_strided_slice_loop_body_stateful_uniform_strided_slice_1_pfor_stridedslice_0Lloop_body/stateful_uniform/Bitcast_1/pfor/while/strided_slice/stack:output:0Nloop_body/stateful_uniform/Bitcast_1/pfor/while/strided_slice/stack_1:output:0Nloop_body/stateful_uniform/Bitcast_1/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
ellipsis_mask*
shrink_axis_mask»
7loop_body/stateful_uniform/Bitcast_1/pfor/while/BitcastBitcastFloop_body/stateful_uniform/Bitcast_1/pfor/while/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
>loop_body/stateful_uniform/Bitcast_1/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ü
:loop_body/stateful_uniform/Bitcast_1/pfor/while/ExpandDims
ExpandDims@loop_body/stateful_uniform/Bitcast_1/pfor/while/Bitcast:output:0Gloop_body/stateful_uniform/Bitcast_1/pfor/while/ExpandDims/dim:output:0*
T0*
_output_shapes

:ê
Tloop_body/stateful_uniform/Bitcast_1/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem=loop_body_stateful_uniform_bitcast_1_pfor_while_placeholder_1;loop_body_stateful_uniform_bitcast_1_pfor_while_placeholderCloop_body/stateful_uniform/Bitcast_1/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0:éèÌy
7loop_body/stateful_uniform/Bitcast_1/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Þ
5loop_body/stateful_uniform/Bitcast_1/pfor/while/add_1AddV2;loop_body_stateful_uniform_bitcast_1_pfor_while_placeholder@loop_body/stateful_uniform/Bitcast_1/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: y
7loop_body/stateful_uniform/Bitcast_1/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :
5loop_body/stateful_uniform/Bitcast_1/pfor/while/add_2AddV2lloop_body_stateful_uniform_bitcast_1_pfor_while_loop_body_stateful_uniform_bitcast_1_pfor_while_loop_counter@loop_body/stateful_uniform/Bitcast_1/pfor/while/add_2/y:output:0*
T0*
_output_shapes
:  
8loop_body/stateful_uniform/Bitcast_1/pfor/while/IdentityIdentity9loop_body/stateful_uniform/Bitcast_1/pfor/while/add_2:z:0*
T0*
_output_shapes
: Û
:loop_body/stateful_uniform/Bitcast_1/pfor/while/Identity_1Identityrloop_body_stateful_uniform_bitcast_1_pfor_while_loop_body_stateful_uniform_bitcast_1_pfor_while_maximum_iterations*
T0*
_output_shapes
: ¢
:loop_body/stateful_uniform/Bitcast_1/pfor/while/Identity_2Identity9loop_body/stateful_uniform/Bitcast_1/pfor/while/add_1:z:0*
T0*
_output_shapes
: Í
:loop_body/stateful_uniform/Bitcast_1/pfor/while/Identity_3Identitydloop_body/stateful_uniform/Bitcast_1/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "}
8loop_body_stateful_uniform_bitcast_1_pfor_while_identityAloop_body/stateful_uniform/Bitcast_1/pfor/while/Identity:output:0"
:loop_body_stateful_uniform_bitcast_1_pfor_while_identity_1Cloop_body/stateful_uniform/Bitcast_1/pfor/while/Identity_1:output:0"
:loop_body_stateful_uniform_bitcast_1_pfor_while_identity_2Cloop_body/stateful_uniform/Bitcast_1/pfor/while/Identity_2:output:0"
:loop_body_stateful_uniform_bitcast_1_pfor_while_identity_3Cloop_body/stateful_uniform/Bitcast_1/pfor/while/Identity_3:output:0"Ô
gloop_body_stateful_uniform_bitcast_1_pfor_while_loop_body_stateful_uniform_bitcast_1_pfor_strided_sliceiloop_body_stateful_uniform_bitcast_1_pfor_while_loop_body_stateful_uniform_bitcast_1_pfor_strided_slice_0"ú
zloop_body_stateful_uniform_bitcast_1_pfor_while_strided_slice_loop_body_stateful_uniform_strided_slice_1_pfor_stridedslice|loop_body_stateful_uniform_bitcast_1_pfor_while_strided_slice_loop_body_stateful_uniform_strided_slice_1_pfor_stridedslice_0*(
_construction_contextkEagerRuntime*0
_input_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ÂX

Cloop_body_transform_ImageProjectiveTransformV3_pfor_while_body_8725
loop_body_transform_imageprojectivetransformv3_pfor_while_loop_body_transform_imageprojectivetransformv3_pfor_while_loop_counter
loop_body_transform_imageprojectivetransformv3_pfor_while_loop_body_transform_imageprojectivetransformv3_pfor_while_maximum_iterationsI
Eloop_body_transform_imageprojectivetransformv3_pfor_while_placeholderK
Gloop_body_transform_imageprojectivetransformv3_pfor_while_placeholder_1
}loop_body_transform_imageprojectivetransformv3_pfor_while_loop_body_transform_imageprojectivetransformv3_pfor_strided_slice_0r
nloop_body_transform_imageprojectivetransformv3_pfor_while_strided_slice_loop_body_expanddims_pfor_expanddims_0~
zloop_body_transform_imageprojectivetransformv3_pfor_while_strided_slice_1_loop_body_rotation_matrix_concat_pfor_concat_1_0a
]loop_body_transform_imageprojectivetransformv3_pfor_while_loop_body_transform_strided_slice_0^
Zloop_body_transform_imageprojectivetransformv3_pfor_while_loop_body_transform_fill_value_0F
Bloop_body_transform_imageprojectivetransformv3_pfor_while_identityH
Dloop_body_transform_imageprojectivetransformv3_pfor_while_identity_1H
Dloop_body_transform_imageprojectivetransformv3_pfor_while_identity_2H
Dloop_body_transform_imageprojectivetransformv3_pfor_while_identity_3
{loop_body_transform_imageprojectivetransformv3_pfor_while_loop_body_transform_imageprojectivetransformv3_pfor_strided_slicep
lloop_body_transform_imageprojectivetransformv3_pfor_while_strided_slice_loop_body_expanddims_pfor_expanddims|
xloop_body_transform_imageprojectivetransformv3_pfor_while_strided_slice_1_loop_body_rotation_matrix_concat_pfor_concat_1_
[loop_body_transform_imageprojectivetransformv3_pfor_while_loop_body_transform_strided_slice\
Xloop_body_transform_imageprojectivetransformv3_pfor_while_loop_body_transform_fill_value
?loop_body/transform/ImageProjectiveTransformV3/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :ø
=loop_body/transform/ImageProjectiveTransformV3/pfor/while/addAddV2Eloop_body_transform_imageprojectivetransformv3_pfor_while_placeholderHloop_body/transform/ImageProjectiveTransformV3/pfor/while/add/y:output:0*
T0*
_output_shapes
: 
Oloop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ¤
Mloop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice/stackPackEloop_body_transform_imageprojectivetransformv3_pfor_while_placeholderXloop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
:
Qloop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ¤
Oloop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice/stack_1PackAloop_body/transform/ImageProjectiveTransformV3/pfor/while/add:z:0Zloop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
: 
Oloop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      À
Gloop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_sliceStridedSlicenloop_body_transform_imageprojectivetransformv3_pfor_while_strided_slice_loop_body_expanddims_pfor_expanddims_0Vloop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice/stack:output:0Xloop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice/stack_1:output:0Xloop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:*
ellipsis_mask*
shrink_axis_mask
Aloop_body/transform/ImageProjectiveTransformV3/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :ü
?loop_body/transform/ImageProjectiveTransformV3/pfor/while/add_1AddV2Eloop_body_transform_imageprojectivetransformv3_pfor_while_placeholderJloop_body/transform/ImageProjectiveTransformV3/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: 
Qloop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ¨
Oloop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice_1/stackPackEloop_body_transform_imageprojectivetransformv3_pfor_while_placeholderZloop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:
Sloop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ª
Qloop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice_1/stack_1PackCloop_body/transform/ImageProjectiveTransformV3/pfor/while/add_1:z:0\loop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:¢
Qloop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ê
Iloop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice_1StridedSlicezloop_body_transform_imageprojectivetransformv3_pfor_while_strided_slice_1_loop_body_rotation_matrix_concat_pfor_concat_1_0Xloop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice_1/stack:output:0Zloop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice_1/stack_1:output:0Zloop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
ellipsis_mask*
shrink_axis_maskÏ
Tloop_body/transform/ImageProjectiveTransformV3/pfor/while/ImageProjectiveTransformV3ImageProjectiveTransformV3Ploop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice:output:0Rloop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice_1:output:0]loop_body_transform_imageprojectivetransformv3_pfor_while_loop_body_transform_strided_slice_0Zloop_body_transform_imageprojectivetransformv3_pfor_while_loop_body_transform_fill_value_0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR
Hloop_body/transform/ImageProjectiveTransformV3/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ×
Dloop_body/transform/ImageProjectiveTransformV3/pfor/while/ExpandDims
ExpandDimsiloop_body/transform/ImageProjectiveTransformV3/pfor/while/ImageProjectiveTransformV3:transformed_images:0Qloop_body/transform/ImageProjectiveTransformV3/pfor/while/ExpandDims/dim:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
^loop_body/transform/ImageProjectiveTransformV3/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemGloop_body_transform_imageprojectivetransformv3_pfor_while_placeholder_1Eloop_body_transform_imageprojectivetransformv3_pfor_while_placeholderMloop_body/transform/ImageProjectiveTransformV3/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0:éèÒ
Aloop_body/transform/ImageProjectiveTransformV3/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :ü
?loop_body/transform/ImageProjectiveTransformV3/pfor/while/add_2AddV2Eloop_body_transform_imageprojectivetransformv3_pfor_while_placeholderJloop_body/transform/ImageProjectiveTransformV3/pfor/while/add_2/y:output:0*
T0*
_output_shapes
: 
Aloop_body/transform/ImageProjectiveTransformV3/pfor/while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :¸
?loop_body/transform/ImageProjectiveTransformV3/pfor/while/add_3AddV2loop_body_transform_imageprojectivetransformv3_pfor_while_loop_body_transform_imageprojectivetransformv3_pfor_while_loop_counterJloop_body/transform/ImageProjectiveTransformV3/pfor/while/add_3/y:output:0*
T0*
_output_shapes
: ´
Bloop_body/transform/ImageProjectiveTransformV3/pfor/while/IdentityIdentityCloop_body/transform/ImageProjectiveTransformV3/pfor/while/add_3:z:0*
T0*
_output_shapes
: ú
Dloop_body/transform/ImageProjectiveTransformV3/pfor/while/Identity_1Identityloop_body_transform_imageprojectivetransformv3_pfor_while_loop_body_transform_imageprojectivetransformv3_pfor_while_maximum_iterations*
T0*
_output_shapes
: ¶
Dloop_body/transform/ImageProjectiveTransformV3/pfor/while/Identity_2IdentityCloop_body/transform/ImageProjectiveTransformV3/pfor/while/add_2:z:0*
T0*
_output_shapes
: á
Dloop_body/transform/ImageProjectiveTransformV3/pfor/while/Identity_3Identitynloop_body/transform/ImageProjectiveTransformV3/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "
Bloop_body_transform_imageprojectivetransformv3_pfor_while_identityKloop_body/transform/ImageProjectiveTransformV3/pfor/while/Identity:output:0"
Dloop_body_transform_imageprojectivetransformv3_pfor_while_identity_1Mloop_body/transform/ImageProjectiveTransformV3/pfor/while/Identity_1:output:0"
Dloop_body_transform_imageprojectivetransformv3_pfor_while_identity_2Mloop_body/transform/ImageProjectiveTransformV3/pfor/while/Identity_2:output:0"
Dloop_body_transform_imageprojectivetransformv3_pfor_while_identity_3Mloop_body/transform/ImageProjectiveTransformV3/pfor/while/Identity_3:output:0"¶
Xloop_body_transform_imageprojectivetransformv3_pfor_while_loop_body_transform_fill_valueZloop_body_transform_imageprojectivetransformv3_pfor_while_loop_body_transform_fill_value_0"ü
{loop_body_transform_imageprojectivetransformv3_pfor_while_loop_body_transform_imageprojectivetransformv3_pfor_strided_slice}loop_body_transform_imageprojectivetransformv3_pfor_while_loop_body_transform_imageprojectivetransformv3_pfor_strided_slice_0"¼
[loop_body_transform_imageprojectivetransformv3_pfor_while_loop_body_transform_strided_slice]loop_body_transform_imageprojectivetransformv3_pfor_while_loop_body_transform_strided_slice_0"ö
xloop_body_transform_imageprojectivetransformv3_pfor_while_strided_slice_1_loop_body_rotation_matrix_concat_pfor_concat_1zloop_body_transform_imageprojectivetransformv3_pfor_while_strided_slice_1_loop_body_rotation_matrix_concat_pfor_concat_1_0"Þ
lloop_body_transform_imageprojectivetransformv3_pfor_while_strided_slice_loop_body_expanddims_pfor_expanddimsnloop_body_transform_imageprojectivetransformv3_pfor_while_strided_slice_loop_body_expanddims_pfor_expanddims_0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :;7
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿ:1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ: 

_output_shapes
::

_output_shapes
: 
Ì-
í
D__inference_sequential_layer_call_and_return_conditional_losses_6662
random_flip_input"
random_rotation_6633:	%
conv2d_6637:
conv2d_6639:'
conv2d_1_6643:0
conv2d_1_6645:0'
conv2d_2_6649:0`
conv2d_2_6651:`

dense_6656:
à

dense_6658:
identity¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dropout/StatefulPartitionedCall¢'random_rotation/StatefulPartitionedCallÑ
random_flip/PartitionedCallPartitionedCallrandom_flip_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_random_flip_layer_call_and_return_conditional_losses_6495
'random_rotation/StatefulPartitionedCallStatefulPartitionedCall$random_flip/PartitionedCall:output:0random_rotation_6633*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_random_rotation_layer_call_and_return_conditional_losses_6423ì
rescaling/PartitionedCallPartitionedCall0random_rotation/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_rescaling_layer_call_and_return_conditional_losses_5321
conv2d/StatefulPartitionedCallStatefulPartitionedCall"rescaling/PartitionedCall:output:0conv2d_6637conv2d_6639*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿüü*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_5334é
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~~* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_5278
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_6643conv2d_1_6645*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿzz0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_5352ï
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ==0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_5290
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_6649conv2d_2_6651*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ99`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_5370Ù
flatten/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿà* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_5382à
dropout/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿà* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_5458
dense/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0
dense_6656
dense_6658*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_5402u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall(^random_rotation/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2R
'random_rotation/StatefulPartitionedCall'random_rotation/StatefulPartitionedCall:d `
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+
_user_specified_namerandom_flip_input
ÿ	
`
A__inference_dropout_layer_call_and_return_conditional_losses_8925

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   Af
dropout/MulMulinputsdropout/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿàC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?¨
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿàq
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿàk
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿà[
IdentityIdentitydropout/Mul_1:z:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿà"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿà:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
 
_user_specified_nameinputs
ñ
_
&__inference_dropout_layer_call_fn_8908

inputs
identity¢StatefulPartitionedCall¾
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿà* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_5458q
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿà`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿà22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
 
_user_specified_nameinputs

û
B__inference_conv2d_1_layer_call_and_return_conditional_losses_8857

inputs8
conv2d_readvariableop_resource:0-
biasadd_readvariableop_resource:0
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿzz0*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿzz0X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿzz0i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿzz0w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ~~: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~~
 
_user_specified_nameinputs
µ

Grandom_rotation_loop_body_stateful_uniform_Bitcast_pfor_while_cond_7053
random_rotation_loop_body_stateful_uniform_bitcast_pfor_while_random_rotation_loop_body_stateful_uniform_bitcast_pfor_while_loop_counter
random_rotation_loop_body_stateful_uniform_bitcast_pfor_while_random_rotation_loop_body_stateful_uniform_bitcast_pfor_while_maximum_iterationsM
Irandom_rotation_loop_body_stateful_uniform_bitcast_pfor_while_placeholderO
Krandom_rotation_loop_body_stateful_uniform_bitcast_pfor_while_placeholder_1
random_rotation_loop_body_stateful_uniform_bitcast_pfor_while_less_random_rotation_loop_body_stateful_uniform_bitcast_pfor_strided_slice£
random_rotation_loop_body_stateful_uniform_bitcast_pfor_while_random_rotation_loop_body_stateful_uniform_bitcast_pfor_while_cond_7053___redundant_placeholder0	J
Frandom_rotation_loop_body_stateful_uniform_bitcast_pfor_while_identity
Á
Brandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/while/LessLessIrandom_rotation_loop_body_stateful_uniform_bitcast_pfor_while_placeholderrandom_rotation_loop_body_stateful_uniform_bitcast_pfor_while_less_random_rotation_loop_body_stateful_uniform_bitcast_pfor_strided_slice*
T0*
_output_shapes
: »
Frandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/while/IdentityIdentityFrandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/while/Less:z:0*
T0
*
_output_shapes
: "
Frandom_rotation_loop_body_stateful_uniform_bitcast_pfor_while_identityOrandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : : : :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
:
öh

Srandom_rotation_loop_body_transform_ImageProjectiveTransformV3_pfor_while_body_7679¥
 random_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_random_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_loop_counter«
¦random_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_random_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_maximum_iterationsY
Urandom_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_placeholder[
Wrandom_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_placeholder_1¢
random_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_random_rotation_loop_body_transform_imageprojectivetransformv3_pfor_strided_slice_0
random_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_strided_slice_random_rotation_loop_body_expanddims_pfor_expanddims_0
random_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_strided_slice_1_random_rotation_loop_body_rotation_matrix_concat_pfor_concat_1_0
}random_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_random_rotation_loop_body_transform_strided_slice_0~
zrandom_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_random_rotation_loop_body_transform_fill_value_0V
Rrandom_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_identityX
Trandom_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_identity_1X
Trandom_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_identity_2X
Trandom_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_identity_3 
random_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_random_rotation_loop_body_transform_imageprojectivetransformv3_pfor_strided_slice
random_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_strided_slice_random_rotation_loop_body_expanddims_pfor_expanddims
random_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_strided_slice_1_random_rotation_loop_body_rotation_matrix_concat_pfor_concat_1
{random_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_random_rotation_loop_body_transform_strided_slice|
xrandom_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_random_rotation_loop_body_transform_fill_value
Orandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :¨
Mrandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/addAddV2Urandom_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_placeholderXrandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/add/y:output:0*
T0*
_output_shapes
: ¡
_random_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : Ô
]random_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice/stackPackUrandom_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_placeholderhrandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
:£
arandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : Ô
_random_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice/stack_1PackQrandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/add:z:0jrandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:°
_random_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¡
Wrandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_sliceStridedSlicerandom_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_strided_slice_random_rotation_loop_body_expanddims_pfor_expanddims_0frandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice/stack:output:0hrandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice/stack_1:output:0hrandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:*
ellipsis_mask*
shrink_axis_mask
Qrandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :¬
Orandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/add_1AddV2Urandom_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_placeholderZrandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: £
arandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : Ø
_random_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice_1/stackPackUrandom_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_placeholderjrandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:¥
crandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : Ú
arandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice_1/stack_1PackSrandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/add_1:z:0lrandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:²
arandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      «
Yrandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice_1StridedSlicerandom_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_strided_slice_1_random_rotation_loop_body_rotation_matrix_concat_pfor_concat_1_0hrandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice_1/stack:output:0jrandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice_1/stack_1:output:0jrandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
ellipsis_mask*
shrink_axis_mask¿
drandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/ImageProjectiveTransformV3ImageProjectiveTransformV3`random_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice:output:0brandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice_1:output:0}random_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_random_rotation_loop_body_transform_strided_slice_0zrandom_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_random_rotation_loop_body_transform_fill_value_0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR
Xrandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
Trandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/ExpandDims
ExpandDimsyrandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/ImageProjectiveTransformV3:transformed_images:0arandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/ExpandDims/dim:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÒ
nrandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemWrandom_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_placeholder_1Urandom_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_placeholder]random_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0:éèÒ
Qrandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :¬
Orandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/add_2AddV2Urandom_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_placeholderZrandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/add_2/y:output:0*
T0*
_output_shapes
: 
Qrandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :ø
Orandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/add_3AddV2 random_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_random_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_loop_counterZrandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/add_3/y:output:0*
T0*
_output_shapes
: Ô
Rrandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/IdentityIdentitySrandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/add_3:z:0*
T0*
_output_shapes
: ª
Trandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/Identity_1Identity¦random_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_random_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_maximum_iterations*
T0*
_output_shapes
: Ö
Trandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/Identity_2IdentitySrandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/add_2:z:0*
T0*
_output_shapes
: 
Trandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/Identity_3Identity~random_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "±
Rrandom_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_identity[random_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/Identity:output:0"µ
Trandom_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_identity_1]random_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/Identity_1:output:0"µ
Trandom_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_identity_2]random_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/Identity_2:output:0"µ
Trandom_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_identity_3]random_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/Identity_3:output:0"ö
xrandom_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_random_rotation_loop_body_transform_fill_valuezrandom_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_random_rotation_loop_body_transform_fill_value_0"¾
random_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_random_rotation_loop_body_transform_imageprojectivetransformv3_pfor_strided_slicerandom_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_random_rotation_loop_body_transform_imageprojectivetransformv3_pfor_strided_slice_0"ü
{random_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_random_rotation_loop_body_transform_strided_slice}random_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_random_rotation_loop_body_transform_strided_slice_0"¸
random_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_strided_slice_1_random_rotation_loop_body_rotation_matrix_concat_pfor_concat_1random_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_strided_slice_1_random_rotation_loop_body_rotation_matrix_concat_pfor_concat_1_0" 
random_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_strided_slice_random_rotation_loop_body_expanddims_pfor_expanddimsrandom_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_strided_slice_random_rotation_loop_body_expanddims_pfor_expanddims_0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :;7
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿ:1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ: 

_output_shapes
::

_output_shapes
: 
*
ò
D__inference_sequential_layer_call_and_return_conditional_losses_5409

inputs%
conv2d_5335:
conv2d_5337:'
conv2d_1_5353:0
conv2d_1_5355:0'
conv2d_2_5371:0`
conv2d_2_5373:`

dense_5403:
à

dense_5405:
identity¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall¢dense/StatefulPartitionedCallÆ
random_flip/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_random_flip_layer_call_and_return_conditional_losses_5304ì
random_rotation/PartitionedCallPartitionedCall$random_flip/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_random_rotation_layer_call_and_return_conditional_losses_5310ä
rescaling/PartitionedCallPartitionedCall(random_rotation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_rescaling_layer_call_and_return_conditional_losses_5321
conv2d/StatefulPartitionedCallStatefulPartitionedCall"rescaling/PartitionedCall:output:0conv2d_5335conv2d_5337*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿüü*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_5334é
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~~* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_5278
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_5353conv2d_1_5355*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿzz0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_5352ï
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ==0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_5290
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_5371conv2d_2_5373*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ99`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_5370Ù
flatten/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿà* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_5382Ð
dropout/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿà* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_5389ø
dense/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0
dense_5403
dense_5405*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_5402u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Å
]
A__inference_flatten_layer_call_and_return_conditional_losses_5382

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`Â ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿàZ
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿà"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ99`:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ99`
 
_user_specified_nameinputs

e
I__inference_random_rotation_layer_call_and_return_conditional_losses_5310

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´

ö
)__inference_sequential_layer_call_fn_6598
random_flip_input
unknown:	#
	unknown_0:
	unknown_1:#
	unknown_2:0
	unknown_3:0#
	unknown_4:0`
	unknown_5:`
	unknown_6:
à
	unknown_7:
identity¢StatefulPartitionedCall¾
StatefulPartitionedCallStatefulPartitionedCallrandom_flip_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_6554o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+
_user_specified_namerandom_flip_input
°
H
,__inference_max_pooling2d_layer_call_fn_8832

inputs
identityÕ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_5278
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ

'__inference_conv2d_1_layer_call_fn_8846

inputs!
unknown:0
	unknown_0:0
identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿzz0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_5352w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿzz0`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ~~: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~~
 
_user_specified_nameinputs
ë
a
E__inference_random_flip_layer_call_and_return_conditional_losses_7865

inputs
identity?
	map/ShapeShapeinputs*
T0*
_output_shapes
:a
map/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
map/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
map/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
map/strided_sliceStridedSlicemap/Shape:output:0 map/strided_slice/stack:output:0"map/strided_slice/stack_1:output:0"map/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
map/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¾
map/TensorArrayV2TensorListReserve(map/TensorArrayV2/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
9map/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*!
valueB"         á
+map/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorinputsBmap/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒK
	map/ConstConst*
_output_shapes
: *
dtype0*
value	B : l
!map/TensorArrayV2_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÂ
map/TensorArrayV2_1TensorListReserve*map/TensorArrayV2_1/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒX
map/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ö
	map/whileStatelessWhilemap/while/loop_counter:output:0map/strided_slice:output:0map/Const:output:0map/TensorArrayV2_1:handle:0map/strided_slice:output:0;map/TensorArrayUnstack/TensorListFromTensor:output_handle:0*
T

2*
_lower_using_switch_merge(*
_num_original_outputs* 
_output_shapes
: : : : : : * 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
map_while_body_7822*
condR
map_while_cond_7821*
output_shapes
: : : : : : 
4map/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*!
valueB"         Ô
&map/TensorArrayV2Stack/TensorListStackTensorListStackmap/while:output:3=map/TensorArrayV2Stack/TensorListStack/element_shape:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
IdentityIdentity/map/TensorArrayV2Stack/TensorListStack:tensor:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ï
J
.__inference_random_rotation_layer_call_fn_7870

inputs
identity¾
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_random_rotation_layer_call_and_return_conditional_losses_5310j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²/
Á
D__inference_sequential_layer_call_and_return_conditional_losses_6777

inputs?
%conv2d_conv2d_readvariableop_resource:4
&conv2d_biasadd_readvariableop_resource:A
'conv2d_1_conv2d_readvariableop_resource:06
(conv2d_1_biasadd_readvariableop_resource:0A
'conv2d_2_conv2d_readvariableop_resource:0`6
(conv2d_2_biasadd_readvariableop_resource:`8
$dense_matmul_readvariableop_resource:
à3
%dense_biasadd_readvariableop_resource:
identity¢conv2d/BiasAdd/ReadVariableOp¢conv2d/Conv2D/ReadVariableOp¢conv2d_1/BiasAdd/ReadVariableOp¢conv2d_1/Conv2D/ReadVariableOp¢conv2d_2/BiasAdd/ReadVariableOp¢conv2d_2/Conv2D/ReadVariableOp¢dense/BiasAdd/ReadVariableOp¢dense/MatMul/ReadVariableOpU
rescaling/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *<]
rescaling/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿe
rescaling/Cast_1Castrescaling/Cast_1/x:output:0*

DstT0*

SrcT0*
_output_shapes
: s
rescaling/mulMulinputsrescaling/Cast/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
rescaling/addAddV2rescaling/mul:z:0rescaling/Cast_1:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0µ
conv2d/Conv2DConv2Drescaling/add:z:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿüü*
paddingVALID*
strides

conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿüüh
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿüü¨
max_pooling2d/MaxPoolMaxPoolconv2d/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~~*
ksize
*
paddingVALID*
strides

conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype0Ä
conv2d_1/Conv2DConv2Dmax_pooling2d/MaxPool:output:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿzz0*
paddingVALID*
strides

conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿzz0j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿzz0¬
max_pooling2d_1/MaxPoolMaxPoolconv2d_1/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ==0*
ksize
*
paddingVALID*
strides

conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:0`*
dtype0Æ
conv2d_2/Conv2DConv2D max_pooling2d_1/MaxPool:output:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ99`*
paddingVALID*
strides

conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ99`j
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ99`^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`Â 
flatten/ReshapeReshapeconv2d_2/Relu:activations:0flatten/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿàj
dropout/IdentityIdentityflatten/Reshape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
à*
dtype0
dense/MatMulMatMuldropout/Identity:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
dense/SoftmaxSoftmaxdense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
IdentityIdentitydense/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô

map_while_cond_6451$
 map_while_map_while_loop_counter
map_while_map_strided_slice
map_while_placeholder
map_while_placeholder_1$
 map_while_less_map_strided_slice:
6map_while_map_while_cond_6451___redundant_placeholder0
map_while_identity
p
map/while/LessLessmap_while_placeholder map_while_less_map_strided_slice*
T0*
_output_shapes
: x
map/while/Less_1Less map_while_map_while_loop_countermap_while_map_strided_slice*
T0*
_output_shapes
: d
map/while/LogicalAnd
LogicalAndmap/while/Less_1:z:0map/while/Less:z:0*
_output_shapes
: Y
map/while/IdentityIdentitymap/while/LogicalAnd:z:0*
T0
*
_output_shapes
: "1
map_while_identitymap/while/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : : : :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
:
ë
a
E__inference_random_flip_layer_call_and_return_conditional_losses_6495

inputs
identity?
	map/ShapeShapeinputs*
T0*
_output_shapes
:a
map/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: c
map/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:c
map/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
map/strided_sliceStridedSlicemap/Shape:output:0 map/strided_slice/stack:output:0"map/strided_slice/stack_1:output:0"map/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
map/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¾
map/TensorArrayV2TensorListReserve(map/TensorArrayV2/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
9map/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*!
valueB"         á
+map/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorinputsBmap/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒK
	map/ConstConst*
_output_shapes
: *
dtype0*
value	B : l
!map/TensorArrayV2_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÂ
map/TensorArrayV2_1TensorListReserve*map/TensorArrayV2_1/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒX
map/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ö
	map/whileStatelessWhilemap/while/loop_counter:output:0map/strided_slice:output:0map/Const:output:0map/TensorArrayV2_1:handle:0map/strided_slice:output:0;map/TensorArrayUnstack/TensorListFromTensor:output_handle:0*
T

2*
_lower_using_switch_merge(*
_num_original_outputs* 
_output_shapes
: : : : : : * 
_read_only_resource_inputs
 *
_stateful_parallelism( *
bodyR
map_while_body_6452*
condR
map_while_cond_6451*
output_shapes
: : : : : : 
4map/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*!
valueB"         Ô
&map/TensorArrayV2Stack/TensorListStackTensorListStackmap/while:output:3=map/TensorArrayV2Stack/TensorListStack/element_shape:output:0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
IdentityIdentity/map/TensorArrayV2Stack/TensorListStack:tensor:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
@
Ç
Nrandom_rotation_loop_body_stateful_uniform_RngReadAndSkip_pfor_while_body_6989
random_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_random_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_loop_counter¡
random_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_random_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_maximum_iterationsT
Prandom_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_placeholderV
Rrandom_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_placeholder_1
random_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_random_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_strided_slice_0
random_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_random_rotation_loop_body_stateful_uniform_rngreadandskip_resource_0:	|
xrandom_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_random_rotation_loop_body_stateful_uniform_cast_x_0|
xrandom_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_random_rotation_loop_body_stateful_uniform_cast_1_0Q
Mrandom_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_identityS
Orandom_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_identity_1S
Orandom_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_identity_2S
Orandom_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_identity_3
random_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_random_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_strided_slice
random_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_random_rotation_loop_body_stateful_uniform_rngreadandskip_resource:	z
vrandom_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_random_rotation_loop_body_stateful_uniform_cast_xz
vrandom_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_random_rotation_loop_body_stateful_uniform_cast_1¢Srandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/while/RngReadAndSkip
Srandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/while/RngReadAndSkipRngReadAndSkiprandom_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_random_rotation_loop_body_stateful_uniform_rngreadandskip_resource_0xrandom_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_random_rotation_loop_body_stateful_uniform_cast_x_0xrandom_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_random_rotation_loop_body_stateful_uniform_cast_1_0*
_output_shapes
:
Srandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : Á
Orandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/while/ExpandDims
ExpandDims[random_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/while/RngReadAndSkip:value:0\random_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/while/ExpandDims/dim:output:0*
T0	*
_output_shapes

:¾
irandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemRrandom_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_placeholder_1Prandom_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_placeholderXrandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0	:éèÐ
Jrandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
Hrandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/while/addAddV2Prandom_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_placeholderSrandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/while/add/y:output:0*
T0*
_output_shapes
: 
Lrandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :ä
Jrandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/while/add_1AddV2random_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_random_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_loop_counterUrandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: 
Mrandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/while/IdentityIdentityNrandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/while/add_1:z:0J^random_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/while/NoOp*
T0*
_output_shapes
: ç
Orandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/while/Identity_1Identityrandom_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_random_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_maximum_iterationsJ^random_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/while/NoOp*
T0*
_output_shapes
: 
Orandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/while/Identity_2IdentityLrandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/while/add:z:0J^random_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/while/NoOp*
T0*
_output_shapes
: Ã
Orandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/while/Identity_3Identityyrandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0J^random_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/while/NoOp*
T0*
_output_shapes
: á
Irandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/while/NoOpNoOpT^random_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/while/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "§
Mrandom_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_identityVrandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/while/Identity:output:0"«
Orandom_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_identity_1Xrandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/while/Identity_1:output:0"«
Orandom_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_identity_2Xrandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/while/Identity_2:output:0"«
Orandom_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_identity_3Xrandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/while/Identity_3:output:0"ò
vrandom_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_random_rotation_loop_body_stateful_uniform_cast_1xrandom_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_random_rotation_loop_body_stateful_uniform_cast_1_0"ò
vrandom_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_random_rotation_loop_body_stateful_uniform_cast_xxrandom_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_random_rotation_loop_body_stateful_uniform_cast_x_0"ª
random_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_random_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_strided_slicerandom_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_random_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_strided_slice_0"
random_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_random_rotation_loop_body_stateful_uniform_rngreadandskip_resourcerandom_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_random_rotation_loop_body_stateful_uniform_rngreadandskip_resource_0*(
_construction_contextkEagerRuntime*#
_input_shapes
: : : : : : : : 2ª
Srandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/while/RngReadAndSkipSrandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/while/RngReadAndSkip: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

û
B__inference_conv2d_1_layer_call_and_return_conditional_losses_5352

inputs8
conv2d_readvariableop_resource:0-
biasadd_readvariableop_resource:0
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿzz0*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:0*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿzz0X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿzz0i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿzz0w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ~~: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~~
 
_user_specified_nameinputs
Ü
_
A__inference_dropout_layer_call_and_return_conditional_losses_5389

inputs

identity_1P
IdentityIdentityinputs*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿà]

Identity_1IdentityIdentity:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿà"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿà:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
 
_user_specified_nameinputs
Ë
È
>loop_body_stateful_uniform_RngReadAndSkip_pfor_while_cond_8034z
vloop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_rngreadandskip_pfor_while_loop_counter
|loop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_rngreadandskip_pfor_while_maximum_iterationsD
@loop_body_stateful_uniform_rngreadandskip_pfor_while_placeholderF
Bloop_body_stateful_uniform_rngreadandskip_pfor_while_placeholder_1z
vloop_body_stateful_uniform_rngreadandskip_pfor_while_less_loop_body_stateful_uniform_rngreadandskip_pfor_strided_slice
loop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_rngreadandskip_pfor_while_cond_8034___redundant_placeholder0
loop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_rngreadandskip_pfor_while_cond_8034___redundant_placeholder1
loop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_rngreadandskip_pfor_while_cond_8034___redundant_placeholder2A
=loop_body_stateful_uniform_rngreadandskip_pfor_while_identity

9loop_body/stateful_uniform/RngReadAndSkip/pfor/while/LessLess@loop_body_stateful_uniform_rngreadandskip_pfor_while_placeholdervloop_body_stateful_uniform_rngreadandskip_pfor_while_less_loop_body_stateful_uniform_rngreadandskip_pfor_strided_slice*
T0*
_output_shapes
: ©
=loop_body/stateful_uniform/RngReadAndSkip/pfor/while/IdentityIdentity=loop_body/stateful_uniform/RngReadAndSkip/pfor/while/Less:z:0*
T0
*
_output_shapes
: "
=loop_body_stateful_uniform_rngreadandskip_pfor_while_identityFloop_body/stateful_uniform/RngReadAndSkip/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
: : : : : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
:

ã
9loop_body_stateful_uniform_Bitcast_1_pfor_while_cond_5796p
lloop_body_stateful_uniform_bitcast_1_pfor_while_loop_body_stateful_uniform_bitcast_1_pfor_while_loop_counterv
rloop_body_stateful_uniform_bitcast_1_pfor_while_loop_body_stateful_uniform_bitcast_1_pfor_while_maximum_iterations?
;loop_body_stateful_uniform_bitcast_1_pfor_while_placeholderA
=loop_body_stateful_uniform_bitcast_1_pfor_while_placeholder_1p
lloop_body_stateful_uniform_bitcast_1_pfor_while_less_loop_body_stateful_uniform_bitcast_1_pfor_strided_slice
loop_body_stateful_uniform_bitcast_1_pfor_while_loop_body_stateful_uniform_bitcast_1_pfor_while_cond_5796___redundant_placeholder0	<
8loop_body_stateful_uniform_bitcast_1_pfor_while_identity

4loop_body/stateful_uniform/Bitcast_1/pfor/while/LessLess;loop_body_stateful_uniform_bitcast_1_pfor_while_placeholderlloop_body_stateful_uniform_bitcast_1_pfor_while_less_loop_body_stateful_uniform_bitcast_1_pfor_strided_slice*
T0*
_output_shapes
: 
8loop_body/stateful_uniform/Bitcast_1/pfor/while/IdentityIdentity8loop_body/stateful_uniform/Bitcast_1/pfor/while/Less:z:0*
T0
*
_output_shapes
: "}
8loop_body_stateful_uniform_bitcast_1_pfor_while_identityAloop_body/stateful_uniform/Bitcast_1/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : : : :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
:
ó]
ú
Hloop_body_stateful_uniform_StatelessRandomUniformV2_pfor_while_body_8224
loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_counter
loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_maximum_iterationsN
Jloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_placeholderP
Lloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_placeholder_1
loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_strided_slice_0
loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_strided_slice_loop_body_stateful_uniform_bitcast_1_pfor_tensorlistconcatv2_0
loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_strided_slice_1_loop_body_stateful_uniform_bitcast_pfor_tensorlistconcatv2_0e
aloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateful_uniform_shape_0|
xloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateful_uniform_statelessrandomuniformv2_alg_0K
Gloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_identityM
Iloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_identity_1M
Iloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_identity_2M
Iloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_identity_3
loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_strided_slice
loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_strided_slice_loop_body_stateful_uniform_bitcast_1_pfor_tensorlistconcatv2
loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_strided_slice_1_loop_body_stateful_uniform_bitcast_pfor_tensorlistconcatv2c
_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateful_uniform_shapez
vloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateful_uniform_statelessrandomuniformv2_alg
Dloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
Bloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/addAddV2Jloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_placeholderMloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/add/y:output:0*
T0*
_output_shapes
: 
Tloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ³
Rloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stackPackJloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_placeholder]loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
:
Vloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ³
Tloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_1PackFloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/add:z:0_loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:¥
Tloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ä
Lloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_sliceStridedSliceloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_strided_slice_loop_body_stateful_uniform_bitcast_1_pfor_tensorlistconcatv2_0[loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack:output:0]loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_1:output:0]loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
ellipsis_mask*
shrink_axis_mask
Floop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
Dloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/add_1AddV2Jloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_placeholderOloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: 
Vloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ·
Tloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stackPackJloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_placeholder_loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:
Xloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ¹
Vloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_1PackHloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/add_1:z:0aloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:§
Vloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ì
Nloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1StridedSliceloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_strided_slice_1_loop_body_stateful_uniform_bitcast_pfor_tensorlistconcatv2_0]loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack:output:0_loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_1:output:0_loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
ellipsis_mask*
shrink_axis_mask¥
Wloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/StatelessRandomUniformV2StatelessRandomUniformV2aloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateful_uniform_shape_0Uloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice:output:0Wloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1:output:0xloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateful_uniform_statelessrandomuniformv2_alg_0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Mloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : Ã
Iloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/ExpandDims
ExpandDims`loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/StatelessRandomUniformV2:output:0Vloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/ExpandDims/dim:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
cloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemLloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_placeholder_1Jloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_placeholderRloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0:éèÒ
Floop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :
Dloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/add_2AddV2Jloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_placeholderOloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/add_2/y:output:0*
T0*
_output_shapes
: 
Floop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :Ì
Dloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/add_3AddV2loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_counterOloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/add_3/y:output:0*
T0*
_output_shapes
: ¾
Gloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/IdentityIdentityHloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/add_3:z:0*
T0*
_output_shapes
: 
Iloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/Identity_1Identityloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_maximum_iterations*
T0*
_output_shapes
: À
Iloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/Identity_2IdentityHloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/add_2:z:0*
T0*
_output_shapes
: ë
Iloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/Identity_3Identitysloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "
Gloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_identityPloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/Identity:output:0"
Iloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_identity_1Rloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/Identity_1:output:0"
Iloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_identity_2Rloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/Identity_2:output:0"
Iloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_identity_3Rloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/Identity_3:output:0"Ä
_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateful_uniform_shapealoop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateful_uniform_shape_0"ò
vloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateful_uniform_statelessrandomuniformv2_algxloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateful_uniform_statelessrandomuniformv2_alg_0"
loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_strided_sliceloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_strided_slice_0"
loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_strided_slice_1_loop_body_stateful_uniform_bitcast_pfor_tensorlistconcatv2loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_strided_slice_1_loop_body_stateful_uniform_bitcast_pfor_tensorlistconcatv2_0"
loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_strided_slice_loop_body_stateful_uniform_bitcast_1_pfor_tensorlistconcatv2loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_strided_slice_loop_body_stateful_uniform_bitcast_1_pfor_tensorlistconcatv2_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ: 

_output_shapes
::

_output_shapes
: 
Å
]
A__inference_flatten_layer_call_and_return_conditional_losses_8898

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`Â ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿàZ
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿà"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ99`:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ99`
 
_user_specified_nameinputs

û
B__inference_conv2d_2_layer_call_and_return_conditional_losses_5370

inputs8
conv2d_readvariableop_resource:0`-
biasadd_readvariableop_resource:`
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0`*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ99`*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ99`X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ99`i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ99`w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ==0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ==0
 
_user_specified_nameinputs
Ç4
	
9loop_body_stateful_uniform_Bitcast_1_pfor_while_body_5797p
lloop_body_stateful_uniform_bitcast_1_pfor_while_loop_body_stateful_uniform_bitcast_1_pfor_while_loop_counterv
rloop_body_stateful_uniform_bitcast_1_pfor_while_loop_body_stateful_uniform_bitcast_1_pfor_while_maximum_iterations?
;loop_body_stateful_uniform_bitcast_1_pfor_while_placeholderA
=loop_body_stateful_uniform_bitcast_1_pfor_while_placeholder_1m
iloop_body_stateful_uniform_bitcast_1_pfor_while_loop_body_stateful_uniform_bitcast_1_pfor_strided_slice_0
|loop_body_stateful_uniform_bitcast_1_pfor_while_strided_slice_loop_body_stateful_uniform_strided_slice_1_pfor_stridedslice_0	<
8loop_body_stateful_uniform_bitcast_1_pfor_while_identity>
:loop_body_stateful_uniform_bitcast_1_pfor_while_identity_1>
:loop_body_stateful_uniform_bitcast_1_pfor_while_identity_2>
:loop_body_stateful_uniform_bitcast_1_pfor_while_identity_3k
gloop_body_stateful_uniform_bitcast_1_pfor_while_loop_body_stateful_uniform_bitcast_1_pfor_strided_slice~
zloop_body_stateful_uniform_bitcast_1_pfor_while_strided_slice_loop_body_stateful_uniform_strided_slice_1_pfor_stridedslice	w
5loop_body/stateful_uniform/Bitcast_1/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ú
3loop_body/stateful_uniform/Bitcast_1/pfor/while/addAddV2;loop_body_stateful_uniform_bitcast_1_pfor_while_placeholder>loop_body/stateful_uniform/Bitcast_1/pfor/while/add/y:output:0*
T0*
_output_shapes
: 
Eloop_body/stateful_uniform/Bitcast_1/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 
Cloop_body/stateful_uniform/Bitcast_1/pfor/while/strided_slice/stackPack;loop_body_stateful_uniform_bitcast_1_pfor_while_placeholderNloop_body/stateful_uniform/Bitcast_1/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
:
Gloop_body/stateful_uniform/Bitcast_1/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : 
Eloop_body/stateful_uniform/Bitcast_1/pfor/while/strided_slice/stack_1Pack7loop_body/stateful_uniform/Bitcast_1/pfor/while/add:z:0Ploop_body/stateful_uniform/Bitcast_1/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:
Eloop_body/stateful_uniform/Bitcast_1/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
=loop_body/stateful_uniform/Bitcast_1/pfor/while/strided_sliceStridedSlice|loop_body_stateful_uniform_bitcast_1_pfor_while_strided_slice_loop_body_stateful_uniform_strided_slice_1_pfor_stridedslice_0Lloop_body/stateful_uniform/Bitcast_1/pfor/while/strided_slice/stack:output:0Nloop_body/stateful_uniform/Bitcast_1/pfor/while/strided_slice/stack_1:output:0Nloop_body/stateful_uniform/Bitcast_1/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
ellipsis_mask*
shrink_axis_mask»
7loop_body/stateful_uniform/Bitcast_1/pfor/while/BitcastBitcastFloop_body/stateful_uniform/Bitcast_1/pfor/while/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
>loop_body/stateful_uniform/Bitcast_1/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ü
:loop_body/stateful_uniform/Bitcast_1/pfor/while/ExpandDims
ExpandDims@loop_body/stateful_uniform/Bitcast_1/pfor/while/Bitcast:output:0Gloop_body/stateful_uniform/Bitcast_1/pfor/while/ExpandDims/dim:output:0*
T0*
_output_shapes

:ê
Tloop_body/stateful_uniform/Bitcast_1/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem=loop_body_stateful_uniform_bitcast_1_pfor_while_placeholder_1;loop_body_stateful_uniform_bitcast_1_pfor_while_placeholderCloop_body/stateful_uniform/Bitcast_1/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0:éèÌy
7loop_body/stateful_uniform/Bitcast_1/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Þ
5loop_body/stateful_uniform/Bitcast_1/pfor/while/add_1AddV2;loop_body_stateful_uniform_bitcast_1_pfor_while_placeholder@loop_body/stateful_uniform/Bitcast_1/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: y
7loop_body/stateful_uniform/Bitcast_1/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :
5loop_body/stateful_uniform/Bitcast_1/pfor/while/add_2AddV2lloop_body_stateful_uniform_bitcast_1_pfor_while_loop_body_stateful_uniform_bitcast_1_pfor_while_loop_counter@loop_body/stateful_uniform/Bitcast_1/pfor/while/add_2/y:output:0*
T0*
_output_shapes
:  
8loop_body/stateful_uniform/Bitcast_1/pfor/while/IdentityIdentity9loop_body/stateful_uniform/Bitcast_1/pfor/while/add_2:z:0*
T0*
_output_shapes
: Û
:loop_body/stateful_uniform/Bitcast_1/pfor/while/Identity_1Identityrloop_body_stateful_uniform_bitcast_1_pfor_while_loop_body_stateful_uniform_bitcast_1_pfor_while_maximum_iterations*
T0*
_output_shapes
: ¢
:loop_body/stateful_uniform/Bitcast_1/pfor/while/Identity_2Identity9loop_body/stateful_uniform/Bitcast_1/pfor/while/add_1:z:0*
T0*
_output_shapes
: Í
:loop_body/stateful_uniform/Bitcast_1/pfor/while/Identity_3Identitydloop_body/stateful_uniform/Bitcast_1/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "}
8loop_body_stateful_uniform_bitcast_1_pfor_while_identityAloop_body/stateful_uniform/Bitcast_1/pfor/while/Identity:output:0"
:loop_body_stateful_uniform_bitcast_1_pfor_while_identity_1Cloop_body/stateful_uniform/Bitcast_1/pfor/while/Identity_1:output:0"
:loop_body_stateful_uniform_bitcast_1_pfor_while_identity_2Cloop_body/stateful_uniform/Bitcast_1/pfor/while/Identity_2:output:0"
:loop_body_stateful_uniform_bitcast_1_pfor_while_identity_3Cloop_body/stateful_uniform/Bitcast_1/pfor/while/Identity_3:output:0"Ô
gloop_body_stateful_uniform_bitcast_1_pfor_while_loop_body_stateful_uniform_bitcast_1_pfor_strided_sliceiloop_body_stateful_uniform_bitcast_1_pfor_while_loop_body_stateful_uniform_bitcast_1_pfor_strided_slice_0"ú
zloop_body_stateful_uniform_bitcast_1_pfor_while_strided_slice_loop_body_stateful_uniform_strided_slice_1_pfor_stridedslice|loop_body_stateful_uniform_bitcast_1_pfor_while_strided_slice_loop_body_stateful_uniform_strided_slice_1_pfor_stridedslice_0*(
_construction_contextkEagerRuntime*0
_input_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
3
ß
7loop_body_stateful_uniform_Bitcast_pfor_while_body_5730l
hloop_body_stateful_uniform_bitcast_pfor_while_loop_body_stateful_uniform_bitcast_pfor_while_loop_counterr
nloop_body_stateful_uniform_bitcast_pfor_while_loop_body_stateful_uniform_bitcast_pfor_while_maximum_iterations=
9loop_body_stateful_uniform_bitcast_pfor_while_placeholder?
;loop_body_stateful_uniform_bitcast_pfor_while_placeholder_1i
eloop_body_stateful_uniform_bitcast_pfor_while_loop_body_stateful_uniform_bitcast_pfor_strided_slice_0|
xloop_body_stateful_uniform_bitcast_pfor_while_strided_slice_loop_body_stateful_uniform_strided_slice_pfor_stridedslice_0	:
6loop_body_stateful_uniform_bitcast_pfor_while_identity<
8loop_body_stateful_uniform_bitcast_pfor_while_identity_1<
8loop_body_stateful_uniform_bitcast_pfor_while_identity_2<
8loop_body_stateful_uniform_bitcast_pfor_while_identity_3g
cloop_body_stateful_uniform_bitcast_pfor_while_loop_body_stateful_uniform_bitcast_pfor_strided_slicez
vloop_body_stateful_uniform_bitcast_pfor_while_strided_slice_loop_body_stateful_uniform_strided_slice_pfor_stridedslice	u
3loop_body/stateful_uniform/Bitcast/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ô
1loop_body/stateful_uniform/Bitcast/pfor/while/addAddV29loop_body_stateful_uniform_bitcast_pfor_while_placeholder<loop_body/stateful_uniform/Bitcast/pfor/while/add/y:output:0*
T0*
_output_shapes
: 
Cloop_body/stateful_uniform/Bitcast/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 
Aloop_body/stateful_uniform/Bitcast/pfor/while/strided_slice/stackPack9loop_body_stateful_uniform_bitcast_pfor_while_placeholderLloop_body/stateful_uniform/Bitcast/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
:
Eloop_body/stateful_uniform/Bitcast/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : 
Cloop_body/stateful_uniform/Bitcast/pfor/while/strided_slice/stack_1Pack5loop_body/stateful_uniform/Bitcast/pfor/while/add:z:0Nloop_body/stateful_uniform/Bitcast/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:
Cloop_body/stateful_uniform/Bitcast/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
;loop_body/stateful_uniform/Bitcast/pfor/while/strided_sliceStridedSlicexloop_body_stateful_uniform_bitcast_pfor_while_strided_slice_loop_body_stateful_uniform_strided_slice_pfor_stridedslice_0Jloop_body/stateful_uniform/Bitcast/pfor/while/strided_slice/stack:output:0Lloop_body/stateful_uniform/Bitcast/pfor/while/strided_slice/stack_1:output:0Lloop_body/stateful_uniform/Bitcast/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
ellipsis_mask*
shrink_axis_mask·
5loop_body/stateful_uniform/Bitcast/pfor/while/BitcastBitcastDloop_body/stateful_uniform/Bitcast/pfor/while/strided_slice:output:0*
T0	*
_output_shapes
:*

type0~
<loop_body/stateful_uniform/Bitcast/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ö
8loop_body/stateful_uniform/Bitcast/pfor/while/ExpandDims
ExpandDims>loop_body/stateful_uniform/Bitcast/pfor/while/Bitcast:output:0Eloop_body/stateful_uniform/Bitcast/pfor/while/ExpandDims/dim:output:0*
T0*
_output_shapes

:â
Rloop_body/stateful_uniform/Bitcast/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem;loop_body_stateful_uniform_bitcast_pfor_while_placeholder_19loop_body_stateful_uniform_bitcast_pfor_while_placeholderAloop_body/stateful_uniform/Bitcast/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0:éèÌw
5loop_body/stateful_uniform/Bitcast/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Ø
3loop_body/stateful_uniform/Bitcast/pfor/while/add_1AddV29loop_body_stateful_uniform_bitcast_pfor_while_placeholder>loop_body/stateful_uniform/Bitcast/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: w
5loop_body/stateful_uniform/Bitcast/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :
3loop_body/stateful_uniform/Bitcast/pfor/while/add_2AddV2hloop_body_stateful_uniform_bitcast_pfor_while_loop_body_stateful_uniform_bitcast_pfor_while_loop_counter>loop_body/stateful_uniform/Bitcast/pfor/while/add_2/y:output:0*
T0*
_output_shapes
: 
6loop_body/stateful_uniform/Bitcast/pfor/while/IdentityIdentity7loop_body/stateful_uniform/Bitcast/pfor/while/add_2:z:0*
T0*
_output_shapes
: Õ
8loop_body/stateful_uniform/Bitcast/pfor/while/Identity_1Identitynloop_body_stateful_uniform_bitcast_pfor_while_loop_body_stateful_uniform_bitcast_pfor_while_maximum_iterations*
T0*
_output_shapes
: 
8loop_body/stateful_uniform/Bitcast/pfor/while/Identity_2Identity7loop_body/stateful_uniform/Bitcast/pfor/while/add_1:z:0*
T0*
_output_shapes
: É
8loop_body/stateful_uniform/Bitcast/pfor/while/Identity_3Identitybloop_body/stateful_uniform/Bitcast/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "y
6loop_body_stateful_uniform_bitcast_pfor_while_identity?loop_body/stateful_uniform/Bitcast/pfor/while/Identity:output:0"}
8loop_body_stateful_uniform_bitcast_pfor_while_identity_1Aloop_body/stateful_uniform/Bitcast/pfor/while/Identity_1:output:0"}
8loop_body_stateful_uniform_bitcast_pfor_while_identity_2Aloop_body/stateful_uniform/Bitcast/pfor/while/Identity_2:output:0"}
8loop_body_stateful_uniform_bitcast_pfor_while_identity_3Aloop_body/stateful_uniform/Bitcast/pfor/while/Identity_3:output:0"Ì
cloop_body_stateful_uniform_bitcast_pfor_while_loop_body_stateful_uniform_bitcast_pfor_strided_sliceeloop_body_stateful_uniform_bitcast_pfor_while_loop_body_stateful_uniform_bitcast_pfor_strided_slice_0"ò
vloop_body_stateful_uniform_bitcast_pfor_while_strided_slice_loop_body_stateful_uniform_strided_slice_pfor_stridedslicexloop_body_stateful_uniform_bitcast_pfor_while_strided_slice_loop_body_stateful_uniform_strided_slice_pfor_stridedslice_0*(
_construction_contextkEagerRuntime*0
_input_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ý>
Á
Irandom_rotation_loop_body_stateful_uniform_Bitcast_1_pfor_while_body_7121
random_rotation_loop_body_stateful_uniform_bitcast_1_pfor_while_random_rotation_loop_body_stateful_uniform_bitcast_1_pfor_while_loop_counter
random_rotation_loop_body_stateful_uniform_bitcast_1_pfor_while_random_rotation_loop_body_stateful_uniform_bitcast_1_pfor_while_maximum_iterationsO
Krandom_rotation_loop_body_stateful_uniform_bitcast_1_pfor_while_placeholderQ
Mrandom_rotation_loop_body_stateful_uniform_bitcast_1_pfor_while_placeholder_1
random_rotation_loop_body_stateful_uniform_bitcast_1_pfor_while_random_rotation_loop_body_stateful_uniform_bitcast_1_pfor_strided_slice_0¡
random_rotation_loop_body_stateful_uniform_bitcast_1_pfor_while_strided_slice_random_rotation_loop_body_stateful_uniform_strided_slice_1_pfor_stridedslice_0	L
Hrandom_rotation_loop_body_stateful_uniform_bitcast_1_pfor_while_identityN
Jrandom_rotation_loop_body_stateful_uniform_bitcast_1_pfor_while_identity_1N
Jrandom_rotation_loop_body_stateful_uniform_bitcast_1_pfor_while_identity_2N
Jrandom_rotation_loop_body_stateful_uniform_bitcast_1_pfor_while_identity_3
random_rotation_loop_body_stateful_uniform_bitcast_1_pfor_while_random_rotation_loop_body_stateful_uniform_bitcast_1_pfor_strided_slice
random_rotation_loop_body_stateful_uniform_bitcast_1_pfor_while_strided_slice_random_rotation_loop_body_stateful_uniform_strided_slice_1_pfor_stridedslice	
Erandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
Crandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/while/addAddV2Krandom_rotation_loop_body_stateful_uniform_bitcast_1_pfor_while_placeholderNrandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/while/add/y:output:0*
T0*
_output_shapes
: 
Urandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ¶
Srandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/while/strided_slice/stackPackKrandom_rotation_loop_body_stateful_uniform_bitcast_1_pfor_while_placeholder^random_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
:
Wrandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ¶
Urandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/while/strided_slice/stack_1PackGrandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/while/add:z:0`random_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:¦
Urandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ù
Mrandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/while/strided_sliceStridedSlicerandom_rotation_loop_body_stateful_uniform_bitcast_1_pfor_while_strided_slice_random_rotation_loop_body_stateful_uniform_strided_slice_1_pfor_stridedslice_0\random_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/while/strided_slice/stack:output:0^random_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/while/strided_slice/stack_1:output:0^random_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
ellipsis_mask*
shrink_axis_maskÛ
Grandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/while/BitcastBitcastVrandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/while/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
Nrandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ¬
Jrandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/while/ExpandDims
ExpandDimsPrandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/while/Bitcast:output:0Wrandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/while/ExpandDims/dim:output:0*
T0*
_output_shapes

:ª
drandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemMrandom_rotation_loop_body_stateful_uniform_bitcast_1_pfor_while_placeholder_1Krandom_rotation_loop_body_stateful_uniform_bitcast_1_pfor_while_placeholderSrandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0:éèÌ
Grandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
Erandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/while/add_1AddV2Krandom_rotation_loop_body_stateful_uniform_bitcast_1_pfor_while_placeholderPrandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: 
Grandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :Ð
Erandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/while/add_2AddV2random_rotation_loop_body_stateful_uniform_bitcast_1_pfor_while_random_rotation_loop_body_stateful_uniform_bitcast_1_pfor_while_loop_counterPrandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/while/add_2/y:output:0*
T0*
_output_shapes
: À
Hrandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/while/IdentityIdentityIrandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/while/add_2:z:0*
T0*
_output_shapes
: 
Jrandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/while/Identity_1Identityrandom_rotation_loop_body_stateful_uniform_bitcast_1_pfor_while_random_rotation_loop_body_stateful_uniform_bitcast_1_pfor_while_maximum_iterations*
T0*
_output_shapes
: Â
Jrandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/while/Identity_2IdentityIrandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/while/add_1:z:0*
T0*
_output_shapes
: í
Jrandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/while/Identity_3Identitytrandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "
Hrandom_rotation_loop_body_stateful_uniform_bitcast_1_pfor_while_identityQrandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/while/Identity:output:0"¡
Jrandom_rotation_loop_body_stateful_uniform_bitcast_1_pfor_while_identity_1Srandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/while/Identity_1:output:0"¡
Jrandom_rotation_loop_body_stateful_uniform_bitcast_1_pfor_while_identity_2Srandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/while/Identity_2:output:0"¡
Jrandom_rotation_loop_body_stateful_uniform_bitcast_1_pfor_while_identity_3Srandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/while/Identity_3:output:0"
random_rotation_loop_body_stateful_uniform_bitcast_1_pfor_while_random_rotation_loop_body_stateful_uniform_bitcast_1_pfor_strided_slicerandom_rotation_loop_body_stateful_uniform_bitcast_1_pfor_while_random_rotation_loop_body_stateful_uniform_bitcast_1_pfor_strided_slice_0"¼
random_rotation_loop_body_stateful_uniform_bitcast_1_pfor_while_strided_slice_random_rotation_loop_body_stateful_uniform_strided_slice_1_pfor_stridedslicerandom_rotation_loop_body_stateful_uniform_bitcast_1_pfor_while_strided_slice_random_rotation_loop_body_stateful_uniform_strided_slice_1_pfor_stridedslice_0*(
_construction_contextkEagerRuntime*0
_input_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ø8
×
__inference__wrapped_model_5269
random_flip_inputJ
0sequential_conv2d_conv2d_readvariableop_resource:?
1sequential_conv2d_biasadd_readvariableop_resource:L
2sequential_conv2d_1_conv2d_readvariableop_resource:0A
3sequential_conv2d_1_biasadd_readvariableop_resource:0L
2sequential_conv2d_2_conv2d_readvariableop_resource:0`A
3sequential_conv2d_2_biasadd_readvariableop_resource:`C
/sequential_dense_matmul_readvariableop_resource:
à>
0sequential_dense_biasadd_readvariableop_resource:
identity¢(sequential/conv2d/BiasAdd/ReadVariableOp¢'sequential/conv2d/Conv2D/ReadVariableOp¢*sequential/conv2d_1/BiasAdd/ReadVariableOp¢)sequential/conv2d_1/Conv2D/ReadVariableOp¢*sequential/conv2d_2/BiasAdd/ReadVariableOp¢)sequential/conv2d_2/Conv2D/ReadVariableOp¢'sequential/dense/BiasAdd/ReadVariableOp¢&sequential/dense/MatMul/ReadVariableOp`
sequential/rescaling/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *<h
sequential/rescaling/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ{
sequential/rescaling/Cast_1Cast&sequential/rescaling/Cast_1/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 
sequential/rescaling/mulMulrandom_flip_input$sequential/rescaling/Cast/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential/rescaling/addAddV2sequential/rescaling/mul:z:0sequential/rescaling/Cast_1:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
'sequential/conv2d/Conv2D/ReadVariableOpReadVariableOp0sequential_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0Ö
sequential/conv2d/Conv2DConv2Dsequential/rescaling/add:z:0/sequential/conv2d/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿüü*
paddingVALID*
strides

(sequential/conv2d/BiasAdd/ReadVariableOpReadVariableOp1sequential_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0µ
sequential/conv2d/BiasAddBiasAdd!sequential/conv2d/Conv2D:output:00sequential/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿüü~
sequential/conv2d/ReluRelu"sequential/conv2d/BiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿüü¾
 sequential/max_pooling2d/MaxPoolMaxPool$sequential/conv2d/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~~*
ksize
*
paddingVALID*
strides
¤
)sequential/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:0*
dtype0å
sequential/conv2d_1/Conv2DConv2D)sequential/max_pooling2d/MaxPool:output:01sequential/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿzz0*
paddingVALID*
strides

*sequential/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:0*
dtype0¹
sequential/conv2d_1/BiasAddBiasAdd#sequential/conv2d_1/Conv2D:output:02sequential/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿzz0
sequential/conv2d_1/ReluRelu$sequential/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿzz0Â
"sequential/max_pooling2d_1/MaxPoolMaxPool&sequential/conv2d_1/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ==0*
ksize
*
paddingVALID*
strides
¤
)sequential/conv2d_2/Conv2D/ReadVariableOpReadVariableOp2sequential_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:0`*
dtype0ç
sequential/conv2d_2/Conv2DConv2D+sequential/max_pooling2d_1/MaxPool:output:01sequential/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ99`*
paddingVALID*
strides

*sequential/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp3sequential_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:`*
dtype0¹
sequential/conv2d_2/BiasAddBiasAdd#sequential/conv2d_2/Conv2D:output:02sequential/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ99`
sequential/conv2d_2/ReluRelu$sequential/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ99`i
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ`Â ¤
sequential/flatten/ReshapeReshape&sequential/conv2d_2/Relu:activations:0!sequential/flatten/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
sequential/dropout/IdentityIdentity#sequential/flatten/Reshape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource* 
_output_shapes
:
à*
dtype0©
sequential/dense/MatMulMatMul$sequential/dropout/Identity:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0©
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
sequential/dense/SoftmaxSoftmax!sequential/dense/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
IdentityIdentity"sequential/dense/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
NoOpNoOp)^sequential/conv2d/BiasAdd/ReadVariableOp(^sequential/conv2d/Conv2D/ReadVariableOp+^sequential/conv2d_1/BiasAdd/ReadVariableOp*^sequential/conv2d_1/Conv2D/ReadVariableOp+^sequential/conv2d_2/BiasAdd/ReadVariableOp*^sequential/conv2d_2/Conv2D/ReadVariableOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2T
(sequential/conv2d/BiasAdd/ReadVariableOp(sequential/conv2d/BiasAdd/ReadVariableOp2R
'sequential/conv2d/Conv2D/ReadVariableOp'sequential/conv2d/Conv2D/ReadVariableOp2X
*sequential/conv2d_1/BiasAdd/ReadVariableOp*sequential/conv2d_1/BiasAdd/ReadVariableOp2V
)sequential/conv2d_1/Conv2D/ReadVariableOp)sequential/conv2d_1/Conv2D/ReadVariableOp2X
*sequential/conv2d_2/BiasAdd/ReadVariableOp*sequential/conv2d_2/BiasAdd/ReadVariableOp2V
)sequential/conv2d_2/Conv2D/ReadVariableOp)sequential/conv2d_2/Conv2D/ReadVariableOp2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp:d `
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+
_user_specified_namerandom_flip_input
­

Hloop_body_stateful_uniform_StatelessRandomUniformV2_pfor_while_cond_8223
loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_counter
loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_maximum_iterationsN
Jloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_placeholderP
Lloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_placeholder_1
loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_less_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_strided_slice¥
 loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_cond_8223___redundant_placeholder0¥
 loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_cond_8223___redundant_placeholder1¥
 loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_cond_8223___redundant_placeholder2¥
 loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_cond_8223___redundant_placeholder3K
Gloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_identity
Å
Cloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/LessLessJloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_placeholderloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_less_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_strided_slice*
T0*
_output_shapes
: ½
Gloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/IdentityIdentityGloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/Less:z:0*
T0
*
_output_shapes
: "
Gloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_identityPloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
: : : : : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
:

ã
9loop_body_stateful_uniform_Bitcast_1_pfor_while_cond_8166p
lloop_body_stateful_uniform_bitcast_1_pfor_while_loop_body_stateful_uniform_bitcast_1_pfor_while_loop_counterv
rloop_body_stateful_uniform_bitcast_1_pfor_while_loop_body_stateful_uniform_bitcast_1_pfor_while_maximum_iterations?
;loop_body_stateful_uniform_bitcast_1_pfor_while_placeholderA
=loop_body_stateful_uniform_bitcast_1_pfor_while_placeholder_1p
lloop_body_stateful_uniform_bitcast_1_pfor_while_less_loop_body_stateful_uniform_bitcast_1_pfor_strided_slice
loop_body_stateful_uniform_bitcast_1_pfor_while_loop_body_stateful_uniform_bitcast_1_pfor_while_cond_8166___redundant_placeholder0	<
8loop_body_stateful_uniform_bitcast_1_pfor_while_identity

4loop_body/stateful_uniform/Bitcast_1/pfor/while/LessLess;loop_body_stateful_uniform_bitcast_1_pfor_while_placeholderlloop_body_stateful_uniform_bitcast_1_pfor_while_less_loop_body_stateful_uniform_bitcast_1_pfor_strided_slice*
T0*
_output_shapes
: 
8loop_body/stateful_uniform/Bitcast_1/pfor/while/IdentityIdentity8loop_body/stateful_uniform/Bitcast_1/pfor/while/Less:z:0*
T0
*
_output_shapes
: "}
8loop_body_stateful_uniform_bitcast_1_pfor_while_identityAloop_body/stateful_uniform/Bitcast_1/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : : : :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
:
ì
Ê
7loop_body_stateful_uniform_Bitcast_pfor_while_cond_5729l
hloop_body_stateful_uniform_bitcast_pfor_while_loop_body_stateful_uniform_bitcast_pfor_while_loop_counterr
nloop_body_stateful_uniform_bitcast_pfor_while_loop_body_stateful_uniform_bitcast_pfor_while_maximum_iterations=
9loop_body_stateful_uniform_bitcast_pfor_while_placeholder?
;loop_body_stateful_uniform_bitcast_pfor_while_placeholder_1l
hloop_body_stateful_uniform_bitcast_pfor_while_less_loop_body_stateful_uniform_bitcast_pfor_strided_slice
~loop_body_stateful_uniform_bitcast_pfor_while_loop_body_stateful_uniform_bitcast_pfor_while_cond_5729___redundant_placeholder0	:
6loop_body_stateful_uniform_bitcast_pfor_while_identity

2loop_body/stateful_uniform/Bitcast/pfor/while/LessLess9loop_body_stateful_uniform_bitcast_pfor_while_placeholderhloop_body_stateful_uniform_bitcast_pfor_while_less_loop_body_stateful_uniform_bitcast_pfor_strided_slice*
T0*
_output_shapes
: 
6loop_body/stateful_uniform/Bitcast/pfor/while/IdentityIdentity6loop_body/stateful_uniform/Bitcast/pfor/while/Less:z:0*
T0
*
_output_shapes
: "y
6loop_body_stateful_uniform_bitcast_pfor_while_identity?loop_body/stateful_uniform/Bitcast/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : : : :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
:

ù
@__inference_conv2d_layer_call_and_return_conditional_losses_5334

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿüü*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿüüZ
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿüük
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿüüw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

e
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_5290

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô

map_while_cond_7821$
 map_while_map_while_loop_counter
map_while_map_strided_slice
map_while_placeholder
map_while_placeholder_1$
 map_while_less_map_strided_slice:
6map_while_map_while_cond_7821___redundant_placeholder0
map_while_identity
p
map/while/LessLessmap_while_placeholder map_while_less_map_strided_slice*
T0*
_output_shapes
: x
map/while/Less_1Less map_while_map_while_loop_countermap_while_map_strided_slice*
T0*
_output_shapes
: d
map/while/LogicalAnd
LogicalAndmap/while/Less_1:z:0map/while/Less:z:0*
_output_shapes
: Y
map/while/IdentityIdentitymap/while/LogicalAnd:z:0*
T0
*
_output_shapes
: "1
map_while_identitymap/while/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : : : :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
:

a
E__inference_random_flip_layer_call_and_return_conditional_losses_7806

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ç
_
C__inference_rescaling_layer_call_and_return_conditional_losses_8807

inputs
identityK
Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *<S
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿQ
Cast_1CastCast_1/x:output:0*

DstT0*

SrcT0*
_output_shapes
: _
mulMulinputsCast/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
addAddV2mul:z:0
Cast_1:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentityadd:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ

map_while_body_6452$
 map_while_map_while_loop_counter
map_while_map_strided_slice
map_while_placeholder
map_while_placeholder_1#
map_while_map_strided_slice_1_0_
[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0
map_while_identity
map_while_identity_1
map_while_identity_2
map_while_identity_3!
map_while_map_strided_slice_1]
Ymap_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor
;map/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ·
-map/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0map_while_placeholderDmap/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*$
_output_shapes
:*
element_dtype0Ü
)map/while/flip_up_down/control_dependencyIdentity4map/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*@
_class6
42loc:@map/while/TensorArrayV2Read/TensorListGetItem*$
_output_shapes
:o
%map/while/flip_up_down/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: À
 map/while/flip_up_down/ReverseV2	ReverseV22map/while/flip_up_down/control_dependency:output:0.map/while/flip_up_down/ReverseV2/axis:output:0*
T0*$
_output_shapes
:Þ
.map/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemmap_while_placeholder_1map_while_placeholder)map/while/flip_up_down/ReverseV2:output:0*
_output_shapes
: *
element_dtype0:éèÒQ
map/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :h
map/while/addAddV2map_while_placeholdermap/while/add/y:output:0*
T0*
_output_shapes
: S
map/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :w
map/while/add_1AddV2 map_while_map_while_loop_countermap/while/add_1/y:output:0*
T0*
_output_shapes
: T
map/while/IdentityIdentitymap/while/add_1:z:0*
T0*
_output_shapes
: ^
map/while/Identity_1Identitymap_while_map_strided_slice*
T0*
_output_shapes
: T
map/while/Identity_2Identitymap/while/add:z:0*
T0*
_output_shapes
: 
map/while/Identity_3Identity>map/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "1
map_while_identitymap/while/Identity:output:0"5
map_while_identity_1map/while/Identity_1:output:0"5
map_while_identity_2map/while/Identity_2:output:0"5
map_while_identity_3map/while/Identity_3:output:0"@
map_while_map_strided_slice_1map_while_map_strided_slice_1_0"¸
Ymap_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

c
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_8837

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
£

ò
?__inference_dense_layer_call_and_return_conditional_losses_8945

inputs2
matmul_readvariableop_resource:
à-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
à*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿà: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
 
_user_specified_nameinputs
ÿ	
`
A__inference_dropout_layer_call_and_return_conditional_losses_5458

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   Af
dropout/MulMulinputsdropout/Const:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿàC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿà*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *fff?¨
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿàq
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿàk
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿà[
IdentityIdentitydropout/Mul_1:z:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿà"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿà:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
 
_user_specified_nameinputs
Ë
Û
Srandom_rotation_loop_body_transform_ImageProjectiveTransformV3_pfor_while_cond_7678¥
 random_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_random_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_loop_counter«
¦random_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_random_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_maximum_iterationsY
Urandom_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_placeholder[
Wrandom_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_placeholder_1¥
 random_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_less_random_rotation_loop_body_transform_imageprojectivetransformv3_pfor_strided_slice»
¶random_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_random_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_cond_7678___redundant_placeholder0»
¶random_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_random_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_cond_7678___redundant_placeholder1»
¶random_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_random_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_cond_7678___redundant_placeholder2»
¶random_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_random_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_cond_7678___redundant_placeholder3V
Rrandom_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_identity
ñ
Nrandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/LessLessUrandom_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_placeholder random_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_less_random_rotation_loop_body_transform_imageprojectivetransformv3_pfor_strided_slice*
T0*
_output_shapes
: Ó
Rrandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/IdentityIdentityRrandom_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/Less:z:0*
T0
*
_output_shapes
: "±
Rrandom_rotation_loop_body_transform_imageprojectivetransformv3_pfor_while_identity[random_rotation/loop_body/transform/ImageProjectiveTransformV3/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
: : : : : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
:
Ä
~
.__inference_random_rotation_layer_call_fn_7877

inputs
unknown:	
identity¢StatefulPartitionedCallØ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_random_rotation_layer_call_and_return_conditional_losses_6423y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 n
ñ
Xrandom_rotation_loop_body_stateful_uniform_StatelessRandomUniformV2_pfor_while_body_7178¯
ªrandom_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_random_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_counterµ
°random_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_random_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_maximum_iterations^
Zrandom_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_placeholder`
\random_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_placeholder_1¬
§random_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_random_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_strided_slice_0°
«random_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_strided_slice_random_rotation_loop_body_stateful_uniform_bitcast_1_pfor_tensorlistconcatv2_0°
«random_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_strided_slice_1_random_rotation_loop_body_stateful_uniform_bitcast_pfor_tensorlistconcatv2_0
random_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_random_rotation_loop_body_stateful_uniform_shape_0
random_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_random_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_alg_0[
Wrandom_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_identity]
Yrandom_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_identity_1]
Yrandom_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_identity_2]
Yrandom_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_identity_3ª
¥random_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_random_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_strided_slice®
©random_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_strided_slice_random_rotation_loop_body_stateful_uniform_bitcast_1_pfor_tensorlistconcatv2®
©random_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_strided_slice_1_random_rotation_loop_body_stateful_uniform_bitcast_pfor_tensorlistconcatv2
random_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_random_rotation_loop_body_stateful_uniform_shape
random_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_random_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_alg
Trandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :·
Rrandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/addAddV2Zrandom_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_placeholder]random_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/add/y:output:0*
T0*
_output_shapes
: ¦
drandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ã
brandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stackPackZrandom_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_placeholdermrandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
:¨
frandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ã
drandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_1PackVrandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/add:z:0orandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:µ
drandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ä
\random_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_sliceStridedSlice«random_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_strided_slice_random_rotation_loop_body_stateful_uniform_bitcast_1_pfor_tensorlistconcatv2_0krandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack:output:0mrandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_1:output:0mrandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
ellipsis_mask*
shrink_axis_mask
Vrandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :»
Trandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/add_1AddV2Zrandom_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_placeholder_random_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: ¨
frandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ç
drandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stackPackZrandom_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_placeholderorandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:ª
hrandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : é
frandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_1PackXrandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/add_1:z:0qrandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:·
frandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ì
^random_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1StridedSlice«random_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_strided_slice_1_random_rotation_loop_body_stateful_uniform_bitcast_pfor_tensorlistconcatv2_0mrandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack:output:0orandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_1:output:0orandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
ellipsis_mask*
shrink_axis_mask
grandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/StatelessRandomUniformV2StatelessRandomUniformV2random_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_random_rotation_loop_body_stateful_uniform_shape_0erandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice:output:0grandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1:output:0random_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_random_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_alg_0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
]random_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ó
Yrandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/ExpandDims
ExpandDimsprandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/StatelessRandomUniformV2:output:0frandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/ExpandDims/dim:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ
srandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem\random_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_placeholder_1Zrandom_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_placeholderbrandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0:éèÒ
Vrandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :»
Trandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/add_2AddV2Zrandom_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_placeholder_random_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/add_2/y:output:0*
T0*
_output_shapes
: 
Vrandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :
Trandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/add_3AddV2ªrandom_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_random_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_counter_random_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/add_3/y:output:0*
T0*
_output_shapes
: Þ
Wrandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/IdentityIdentityXrandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/add_3:z:0*
T0*
_output_shapes
: ¹
Yrandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/Identity_1Identity°random_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_random_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_maximum_iterations*
T0*
_output_shapes
: à
Yrandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/Identity_2IdentityXrandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/add_2:z:0*
T0*
_output_shapes
: 
Yrandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/Identity_3Identityrandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "»
Wrandom_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_identity`random_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/Identity:output:0"¿
Yrandom_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_identity_1brandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/Identity_1:output:0"¿
Yrandom_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_identity_2brandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/Identity_2:output:0"¿
Yrandom_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_identity_3brandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/Identity_3:output:0"
random_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_random_rotation_loop_body_stateful_uniform_shaperandom_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_random_rotation_loop_body_stateful_uniform_shape_0"´
random_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_random_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_algrandom_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_random_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_alg_0"Ò
¥random_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_random_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_strided_slice§random_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_random_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_strided_slice_0"Ú
©random_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_strided_slice_1_random_rotation_loop_body_stateful_uniform_bitcast_pfor_tensorlistconcatv2«random_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_strided_slice_1_random_rotation_loop_body_stateful_uniform_bitcast_pfor_tensorlistconcatv2_0"Ú
©random_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_strided_slice_random_rotation_loop_body_stateful_uniform_bitcast_1_pfor_tensorlistconcatv2«random_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_strided_slice_random_rotation_loop_body_stateful_uniform_bitcast_1_pfor_tensorlistconcatv2_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ: 

_output_shapes
::

_output_shapes
: 
æ

'__inference_conv2d_2_layer_call_fn_8876

inputs!
unknown:0`
	unknown_0:`
identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ99`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_5370w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ99``
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ==0: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ==0
 
_user_specified_nameinputs
Í
µ
Xrandom_rotation_loop_body_stateful_uniform_StatelessRandomUniformV2_pfor_while_cond_7177¯
ªrandom_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_random_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_counterµ
°random_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_random_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_maximum_iterations^
Zrandom_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_placeholder`
\random_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_placeholder_1¯
ªrandom_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_less_random_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_strided_sliceÅ
Àrandom_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_random_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_cond_7177___redundant_placeholder0Å
Àrandom_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_random_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_cond_7177___redundant_placeholder1Å
Àrandom_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_random_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_cond_7177___redundant_placeholder2Å
Àrandom_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_random_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_cond_7177___redundant_placeholder3[
Wrandom_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_identity

Srandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/LessLessZrandom_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_placeholderªrandom_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_less_random_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_strided_slice*
T0*
_output_shapes
: Ý
Wrandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/IdentityIdentityWrandom_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/Less:z:0*
T0
*
_output_shapes
: "»
Wrandom_rotation_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_identity`random_rotation/loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
: : : : : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
:
3
ß
7loop_body_stateful_uniform_Bitcast_pfor_while_body_8100l
hloop_body_stateful_uniform_bitcast_pfor_while_loop_body_stateful_uniform_bitcast_pfor_while_loop_counterr
nloop_body_stateful_uniform_bitcast_pfor_while_loop_body_stateful_uniform_bitcast_pfor_while_maximum_iterations=
9loop_body_stateful_uniform_bitcast_pfor_while_placeholder?
;loop_body_stateful_uniform_bitcast_pfor_while_placeholder_1i
eloop_body_stateful_uniform_bitcast_pfor_while_loop_body_stateful_uniform_bitcast_pfor_strided_slice_0|
xloop_body_stateful_uniform_bitcast_pfor_while_strided_slice_loop_body_stateful_uniform_strided_slice_pfor_stridedslice_0	:
6loop_body_stateful_uniform_bitcast_pfor_while_identity<
8loop_body_stateful_uniform_bitcast_pfor_while_identity_1<
8loop_body_stateful_uniform_bitcast_pfor_while_identity_2<
8loop_body_stateful_uniform_bitcast_pfor_while_identity_3g
cloop_body_stateful_uniform_bitcast_pfor_while_loop_body_stateful_uniform_bitcast_pfor_strided_slicez
vloop_body_stateful_uniform_bitcast_pfor_while_strided_slice_loop_body_stateful_uniform_strided_slice_pfor_stridedslice	u
3loop_body/stateful_uniform/Bitcast/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ô
1loop_body/stateful_uniform/Bitcast/pfor/while/addAddV29loop_body_stateful_uniform_bitcast_pfor_while_placeholder<loop_body/stateful_uniform/Bitcast/pfor/while/add/y:output:0*
T0*
_output_shapes
: 
Cloop_body/stateful_uniform/Bitcast/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 
Aloop_body/stateful_uniform/Bitcast/pfor/while/strided_slice/stackPack9loop_body_stateful_uniform_bitcast_pfor_while_placeholderLloop_body/stateful_uniform/Bitcast/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
:
Eloop_body/stateful_uniform/Bitcast/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : 
Cloop_body/stateful_uniform/Bitcast/pfor/while/strided_slice/stack_1Pack5loop_body/stateful_uniform/Bitcast/pfor/while/add:z:0Nloop_body/stateful_uniform/Bitcast/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:
Cloop_body/stateful_uniform/Bitcast/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
;loop_body/stateful_uniform/Bitcast/pfor/while/strided_sliceStridedSlicexloop_body_stateful_uniform_bitcast_pfor_while_strided_slice_loop_body_stateful_uniform_strided_slice_pfor_stridedslice_0Jloop_body/stateful_uniform/Bitcast/pfor/while/strided_slice/stack:output:0Lloop_body/stateful_uniform/Bitcast/pfor/while/strided_slice/stack_1:output:0Lloop_body/stateful_uniform/Bitcast/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
ellipsis_mask*
shrink_axis_mask·
5loop_body/stateful_uniform/Bitcast/pfor/while/BitcastBitcastDloop_body/stateful_uniform/Bitcast/pfor/while/strided_slice:output:0*
T0	*
_output_shapes
:*

type0~
<loop_body/stateful_uniform/Bitcast/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ö
8loop_body/stateful_uniform/Bitcast/pfor/while/ExpandDims
ExpandDims>loop_body/stateful_uniform/Bitcast/pfor/while/Bitcast:output:0Eloop_body/stateful_uniform/Bitcast/pfor/while/ExpandDims/dim:output:0*
T0*
_output_shapes

:â
Rloop_body/stateful_uniform/Bitcast/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem;loop_body_stateful_uniform_bitcast_pfor_while_placeholder_19loop_body_stateful_uniform_bitcast_pfor_while_placeholderAloop_body/stateful_uniform/Bitcast/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0:éèÌw
5loop_body/stateful_uniform/Bitcast/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Ø
3loop_body/stateful_uniform/Bitcast/pfor/while/add_1AddV29loop_body_stateful_uniform_bitcast_pfor_while_placeholder>loop_body/stateful_uniform/Bitcast/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: w
5loop_body/stateful_uniform/Bitcast/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :
3loop_body/stateful_uniform/Bitcast/pfor/while/add_2AddV2hloop_body_stateful_uniform_bitcast_pfor_while_loop_body_stateful_uniform_bitcast_pfor_while_loop_counter>loop_body/stateful_uniform/Bitcast/pfor/while/add_2/y:output:0*
T0*
_output_shapes
: 
6loop_body/stateful_uniform/Bitcast/pfor/while/IdentityIdentity7loop_body/stateful_uniform/Bitcast/pfor/while/add_2:z:0*
T0*
_output_shapes
: Õ
8loop_body/stateful_uniform/Bitcast/pfor/while/Identity_1Identitynloop_body_stateful_uniform_bitcast_pfor_while_loop_body_stateful_uniform_bitcast_pfor_while_maximum_iterations*
T0*
_output_shapes
: 
8loop_body/stateful_uniform/Bitcast/pfor/while/Identity_2Identity7loop_body/stateful_uniform/Bitcast/pfor/while/add_1:z:0*
T0*
_output_shapes
: É
8loop_body/stateful_uniform/Bitcast/pfor/while/Identity_3Identitybloop_body/stateful_uniform/Bitcast/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "y
6loop_body_stateful_uniform_bitcast_pfor_while_identity?loop_body/stateful_uniform/Bitcast/pfor/while/Identity:output:0"}
8loop_body_stateful_uniform_bitcast_pfor_while_identity_1Aloop_body/stateful_uniform/Bitcast/pfor/while/Identity_1:output:0"}
8loop_body_stateful_uniform_bitcast_pfor_while_identity_2Aloop_body/stateful_uniform/Bitcast/pfor/while/Identity_2:output:0"}
8loop_body_stateful_uniform_bitcast_pfor_while_identity_3Aloop_body/stateful_uniform/Bitcast/pfor/while/Identity_3:output:0"Ì
cloop_body_stateful_uniform_bitcast_pfor_while_loop_body_stateful_uniform_bitcast_pfor_strided_sliceeloop_body_stateful_uniform_bitcast_pfor_while_loop_body_stateful_uniform_bitcast_pfor_strided_slice_0"ò
vloop_body_stateful_uniform_bitcast_pfor_while_strided_slice_loop_body_stateful_uniform_strided_slice_pfor_stridedslicexloop_body_stateful_uniform_bitcast_pfor_while_strided_slice_loop_body_stateful_uniform_strided_slice_pfor_stridedslice_0*(
_construction_contextkEagerRuntime*0
_input_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

a
E__inference_random_flip_layer_call_and_return_conditional_losses_5304

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ì	
Ò
)__inference_sequential_layer_call_fn_6712

inputs!
unknown:
	unknown_0:#
	unknown_1:0
	unknown_2:0#
	unknown_3:0`
	unknown_4:`
	unknown_5:
à
	unknown_6:
identity¢StatefulPartitionedCall§
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_5409o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
á	
Ö
"__inference_signature_wrapper_6691
random_flip_input!
unknown:
	unknown_0:#
	unknown_1:0
	unknown_2:0#
	unknown_3:0`
	unknown_4:`
	unknown_5:
à
	unknown_6:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallrandom_flip_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__wrapped_model_5269o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+
_user_specified_namerandom_flip_input

e
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_8867

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾

$__inference_dense_layer_call_fn_8934

inputs
unknown:
à
	unknown_0:
identity¢StatefulPartitionedCallÔ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_5402o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿà: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
 
_user_specified_nameinputs
Ë
È
>loop_body_stateful_uniform_RngReadAndSkip_pfor_while_cond_5664z
vloop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_rngreadandskip_pfor_while_loop_counter
|loop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_rngreadandskip_pfor_while_maximum_iterationsD
@loop_body_stateful_uniform_rngreadandskip_pfor_while_placeholderF
Bloop_body_stateful_uniform_rngreadandskip_pfor_while_placeholder_1z
vloop_body_stateful_uniform_rngreadandskip_pfor_while_less_loop_body_stateful_uniform_rngreadandskip_pfor_strided_slice
loop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_rngreadandskip_pfor_while_cond_5664___redundant_placeholder0
loop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_rngreadandskip_pfor_while_cond_5664___redundant_placeholder1
loop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_rngreadandskip_pfor_while_cond_5664___redundant_placeholder2A
=loop_body_stateful_uniform_rngreadandskip_pfor_while_identity

9loop_body/stateful_uniform/RngReadAndSkip/pfor/while/LessLess@loop_body_stateful_uniform_rngreadandskip_pfor_while_placeholdervloop_body_stateful_uniform_rngreadandskip_pfor_while_less_loop_body_stateful_uniform_rngreadandskip_pfor_strided_slice*
T0*
_output_shapes
: ©
=loop_body/stateful_uniform/RngReadAndSkip/pfor/while/IdentityIdentity=loop_body/stateful_uniform/RngReadAndSkip/pfor/while/Less:z:0*
T0
*
_output_shapes
: "
=loop_body_stateful_uniform_rngreadandskip_pfor_while_identityFloop_body/stateful_uniform/RngReadAndSkip/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
: : : : : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
:
ì
Ê
7loop_body_stateful_uniform_Bitcast_pfor_while_cond_8099l
hloop_body_stateful_uniform_bitcast_pfor_while_loop_body_stateful_uniform_bitcast_pfor_while_loop_counterr
nloop_body_stateful_uniform_bitcast_pfor_while_loop_body_stateful_uniform_bitcast_pfor_while_maximum_iterations=
9loop_body_stateful_uniform_bitcast_pfor_while_placeholder?
;loop_body_stateful_uniform_bitcast_pfor_while_placeholder_1l
hloop_body_stateful_uniform_bitcast_pfor_while_less_loop_body_stateful_uniform_bitcast_pfor_strided_slice
~loop_body_stateful_uniform_bitcast_pfor_while_loop_body_stateful_uniform_bitcast_pfor_while_cond_8099___redundant_placeholder0	:
6loop_body_stateful_uniform_bitcast_pfor_while_identity

2loop_body/stateful_uniform/Bitcast/pfor/while/LessLess9loop_body_stateful_uniform_bitcast_pfor_while_placeholderhloop_body_stateful_uniform_bitcast_pfor_while_less_loop_body_stateful_uniform_bitcast_pfor_strided_slice*
T0*
_output_shapes
: 
6loop_body/stateful_uniform/Bitcast/pfor/while/IdentityIdentity6loop_body/stateful_uniform/Bitcast/pfor/while/Less:z:0*
T0
*
_output_shapes
: "y
6loop_body_stateful_uniform_bitcast_pfor_while_identity?loop_body/stateful_uniform/Bitcast/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : : : :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
:

B
&__inference_dropout_layer_call_fn_8903

inputs
identity®
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿà* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_5389b
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿà"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿà:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
 
_user_specified_nameinputs
£

ò
?__inference_dense_layer_call_and_return_conditional_losses_5402

inputs2
matmul_readvariableop_resource:
à-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
à*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:ÿÿÿÿÿÿÿÿÿà: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
 
_user_specified_nameinputs
Ü
_
A__inference_dropout_layer_call_and_return_conditional_losses_8913

inputs

identity_1P
IdentityIdentityinputs*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿà]

Identity_1IdentityIdentity:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿà"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿà:Q M
)
_output_shapes
:ÿÿÿÿÿÿÿÿÿà
 
_user_specified_nameinputs
µ

map_while_body_7822$
 map_while_map_while_loop_counter
map_while_map_strided_slice
map_while_placeholder
map_while_placeholder_1#
map_while_map_strided_slice_1_0_
[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0
map_while_identity
map_while_identity_1
map_while_identity_2
map_while_identity_3!
map_while_map_strided_slice_1]
Ymap_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor
;map/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ·
-map/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0map_while_placeholderDmap/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*$
_output_shapes
:*
element_dtype0Ü
)map/while/flip_up_down/control_dependencyIdentity4map/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*@
_class6
42loc:@map/while/TensorArrayV2Read/TensorListGetItem*$
_output_shapes
:o
%map/while/flip_up_down/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: À
 map/while/flip_up_down/ReverseV2	ReverseV22map/while/flip_up_down/control_dependency:output:0.map/while/flip_up_down/ReverseV2/axis:output:0*
T0*$
_output_shapes
:Þ
.map/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemmap_while_placeholder_1map_while_placeholder)map/while/flip_up_down/ReverseV2:output:0*
_output_shapes
: *
element_dtype0:éèÒQ
map/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :h
map/while/addAddV2map_while_placeholdermap/while/add/y:output:0*
T0*
_output_shapes
: S
map/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :w
map/while/add_1AddV2 map_while_map_while_loop_countermap/while/add_1/y:output:0*
T0*
_output_shapes
: T
map/while/IdentityIdentitymap/while/add_1:z:0*
T0*
_output_shapes
: ^
map/while/Identity_1Identitymap_while_map_strided_slice*
T0*
_output_shapes
: T
map/while/Identity_2Identitymap/while/add:z:0*
T0*
_output_shapes
: 
map/while/Identity_3Identity>map/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "1
map_while_identitymap/while/Identity:output:0"5
map_while_identity_1map/while/Identity_1:output:0"5
map_while_identity_2map/while/Identity_2:output:0"5
map_while_identity_3map/while/Identity_3:output:0"@
map_while_map_strided_slice_1map_while_map_strided_slice_1_0"¸
Ymap_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ó]
ú
Hloop_body_stateful_uniform_StatelessRandomUniformV2_pfor_while_body_5854
loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_counter
loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_maximum_iterationsN
Jloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_placeholderP
Lloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_placeholder_1
loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_strided_slice_0
loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_strided_slice_loop_body_stateful_uniform_bitcast_1_pfor_tensorlistconcatv2_0
loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_strided_slice_1_loop_body_stateful_uniform_bitcast_pfor_tensorlistconcatv2_0e
aloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateful_uniform_shape_0|
xloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateful_uniform_statelessrandomuniformv2_alg_0K
Gloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_identityM
Iloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_identity_1M
Iloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_identity_2M
Iloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_identity_3
loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_strided_slice
loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_strided_slice_loop_body_stateful_uniform_bitcast_1_pfor_tensorlistconcatv2
loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_strided_slice_1_loop_body_stateful_uniform_bitcast_pfor_tensorlistconcatv2c
_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateful_uniform_shapez
vloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateful_uniform_statelessrandomuniformv2_alg
Dloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
Bloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/addAddV2Jloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_placeholderMloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/add/y:output:0*
T0*
_output_shapes
: 
Tloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ³
Rloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stackPackJloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_placeholder]loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
:
Vloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ³
Tloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_1PackFloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/add:z:0_loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:¥
Tloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ä
Lloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_sliceStridedSliceloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_strided_slice_loop_body_stateful_uniform_bitcast_1_pfor_tensorlistconcatv2_0[loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack:output:0]loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_1:output:0]loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
ellipsis_mask*
shrink_axis_mask
Floop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
Dloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/add_1AddV2Jloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_placeholderOloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: 
Vloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ·
Tloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stackPackJloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_placeholder_loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:
Xloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ¹
Vloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_1PackHloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/add_1:z:0aloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:§
Vloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ì
Nloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1StridedSliceloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_strided_slice_1_loop_body_stateful_uniform_bitcast_pfor_tensorlistconcatv2_0]loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack:output:0_loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_1:output:0_loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
ellipsis_mask*
shrink_axis_mask¥
Wloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/StatelessRandomUniformV2StatelessRandomUniformV2aloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateful_uniform_shape_0Uloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice:output:0Wloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/strided_slice_1:output:0xloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateful_uniform_statelessrandomuniformv2_alg_0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Mloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : Ã
Iloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/ExpandDims
ExpandDims`loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/StatelessRandomUniformV2:output:0Vloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/ExpandDims/dim:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
cloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemLloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_placeholder_1Jloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_placeholderRloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0:éèÒ
Floop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :
Dloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/add_2AddV2Jloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_placeholderOloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/add_2/y:output:0*
T0*
_output_shapes
: 
Floop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :Ì
Dloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/add_3AddV2loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_counterOloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/add_3/y:output:0*
T0*
_output_shapes
: ¾
Gloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/IdentityIdentityHloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/add_3:z:0*
T0*
_output_shapes
: 
Iloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/Identity_1Identityloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_maximum_iterations*
T0*
_output_shapes
: À
Iloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/Identity_2IdentityHloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/add_2:z:0*
T0*
_output_shapes
: ë
Iloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/Identity_3Identitysloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "
Gloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_identityPloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/Identity:output:0"
Iloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_identity_1Rloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/Identity_1:output:0"
Iloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_identity_2Rloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/Identity_2:output:0"
Iloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_identity_3Rloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/Identity_3:output:0"Ä
_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateful_uniform_shapealoop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateful_uniform_shape_0"ò
vloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateful_uniform_statelessrandomuniformv2_algxloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateful_uniform_statelessrandomuniformv2_alg_0"
loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_strided_sliceloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_strided_slice_0"
loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_strided_slice_1_loop_body_stateful_uniform_bitcast_pfor_tensorlistconcatv2loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_strided_slice_1_loop_body_stateful_uniform_bitcast_pfor_tensorlistconcatv2_0"
loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_strided_slice_loop_body_stateful_uniform_bitcast_1_pfor_tensorlistconcatv2loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_strided_slice_loop_body_stateful_uniform_bitcast_1_pfor_tensorlistconcatv2_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ: 

_output_shapes
::

_output_shapes
: 
´
J
.__inference_max_pooling2d_1_layer_call_fn_8862

inputs
identity×
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_5290
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«
B
&__inference_flatten_layer_call_fn_8892

inputs
identity®
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿà* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_5382b
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:ÿÿÿÿÿÿÿÿÿà"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ99`:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ99`
 
_user_specified_nameinputs
¯*
ý
D__inference_sequential_layer_call_and_return_conditional_losses_6629
random_flip_input%
conv2d_6604:
conv2d_6606:'
conv2d_1_6610:0
conv2d_1_6612:0'
conv2d_2_6616:0`
conv2d_2_6618:`

dense_6623:
à

dense_6625:
identity¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall¢dense/StatefulPartitionedCallÑ
random_flip/PartitionedCallPartitionedCallrandom_flip_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_random_flip_layer_call_and_return_conditional_losses_5304ì
random_rotation/PartitionedCallPartitionedCall$random_flip/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_random_rotation_layer_call_and_return_conditional_losses_5310ä
rescaling/PartitionedCallPartitionedCall(random_rotation/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_rescaling_layer_call_and_return_conditional_losses_5321
conv2d/StatefulPartitionedCallStatefulPartitionedCall"rescaling/PartitionedCall:output:0conv2d_6604conv2d_6606*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿüü*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_5334é
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~~* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_5278
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_6610conv2d_1_6612*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿzz0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_5352ï
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ==0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_5290
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_6616conv2d_2_6618*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ99`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_5370Ù
flatten/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿà* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_5382Ð
dropout/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿà* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_5389ø
dense/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0
dense_6623
dense_6625*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_5402u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:d `
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+
_user_specified_namerandom_flip_input

e
I__inference_random_rotation_layer_call_and_return_conditional_losses_7881

inputs
identityX
IdentityIdentityinputs*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¢4
ü
>loop_body_stateful_uniform_RngReadAndSkip_pfor_while_body_5665z
vloop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_rngreadandskip_pfor_while_loop_counter
|loop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_rngreadandskip_pfor_while_maximum_iterationsD
@loop_body_stateful_uniform_rngreadandskip_pfor_while_placeholderF
Bloop_body_stateful_uniform_rngreadandskip_pfor_while_placeholder_1w
sloop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_rngreadandskip_pfor_strided_slice_0w
iloop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_rngreadandskip_resource_0:	\
Xloop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_cast_x_0\
Xloop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_cast_1_0A
=loop_body_stateful_uniform_rngreadandskip_pfor_while_identityC
?loop_body_stateful_uniform_rngreadandskip_pfor_while_identity_1C
?loop_body_stateful_uniform_rngreadandskip_pfor_while_identity_2C
?loop_body_stateful_uniform_rngreadandskip_pfor_while_identity_3u
qloop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_rngreadandskip_pfor_strided_sliceu
gloop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_rngreadandskip_resource:	Z
Vloop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_cast_xZ
Vloop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_cast_1¢Cloop_body/stateful_uniform/RngReadAndSkip/pfor/while/RngReadAndSkip
Cloop_body/stateful_uniform/RngReadAndSkip/pfor/while/RngReadAndSkipRngReadAndSkipiloop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_rngreadandskip_resource_0Xloop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_cast_x_0Xloop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_cast_1_0*
_output_shapes
:
Cloop_body/stateful_uniform/RngReadAndSkip/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
?loop_body/stateful_uniform/RngReadAndSkip/pfor/while/ExpandDims
ExpandDimsKloop_body/stateful_uniform/RngReadAndSkip/pfor/while/RngReadAndSkip:value:0Lloop_body/stateful_uniform/RngReadAndSkip/pfor/while/ExpandDims/dim:output:0*
T0	*
_output_shapes

:þ
Yloop_body/stateful_uniform/RngReadAndSkip/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemBloop_body_stateful_uniform_rngreadandskip_pfor_while_placeholder_1@loop_body_stateful_uniform_rngreadandskip_pfor_while_placeholderHloop_body/stateful_uniform/RngReadAndSkip/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0	:éèÐ|
:loop_body/stateful_uniform/RngReadAndSkip/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :é
8loop_body/stateful_uniform/RngReadAndSkip/pfor/while/addAddV2@loop_body_stateful_uniform_rngreadandskip_pfor_while_placeholderCloop_body/stateful_uniform/RngReadAndSkip/pfor/while/add/y:output:0*
T0*
_output_shapes
: ~
<loop_body/stateful_uniform/RngReadAndSkip/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :£
:loop_body/stateful_uniform/RngReadAndSkip/pfor/while/add_1AddV2vloop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_rngreadandskip_pfor_while_loop_counterEloop_body/stateful_uniform/RngReadAndSkip/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: æ
=loop_body/stateful_uniform/RngReadAndSkip/pfor/while/IdentityIdentity>loop_body/stateful_uniform/RngReadAndSkip/pfor/while/add_1:z:0:^loop_body/stateful_uniform/RngReadAndSkip/pfor/while/NoOp*
T0*
_output_shapes
: ¦
?loop_body/stateful_uniform/RngReadAndSkip/pfor/while/Identity_1Identity|loop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_rngreadandskip_pfor_while_maximum_iterations:^loop_body/stateful_uniform/RngReadAndSkip/pfor/while/NoOp*
T0*
_output_shapes
: æ
?loop_body/stateful_uniform/RngReadAndSkip/pfor/while/Identity_2Identity<loop_body/stateful_uniform/RngReadAndSkip/pfor/while/add:z:0:^loop_body/stateful_uniform/RngReadAndSkip/pfor/while/NoOp*
T0*
_output_shapes
: 
?loop_body/stateful_uniform/RngReadAndSkip/pfor/while/Identity_3Identityiloop_body/stateful_uniform/RngReadAndSkip/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0:^loop_body/stateful_uniform/RngReadAndSkip/pfor/while/NoOp*
T0*
_output_shapes
: Á
9loop_body/stateful_uniform/RngReadAndSkip/pfor/while/NoOpNoOpD^loop_body/stateful_uniform/RngReadAndSkip/pfor/while/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
=loop_body_stateful_uniform_rngreadandskip_pfor_while_identityFloop_body/stateful_uniform/RngReadAndSkip/pfor/while/Identity:output:0"
?loop_body_stateful_uniform_rngreadandskip_pfor_while_identity_1Hloop_body/stateful_uniform/RngReadAndSkip/pfor/while/Identity_1:output:0"
?loop_body_stateful_uniform_rngreadandskip_pfor_while_identity_2Hloop_body/stateful_uniform/RngReadAndSkip/pfor/while/Identity_2:output:0"
?loop_body_stateful_uniform_rngreadandskip_pfor_while_identity_3Hloop_body/stateful_uniform/RngReadAndSkip/pfor/while/Identity_3:output:0"²
Vloop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_cast_1Xloop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_cast_1_0"²
Vloop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_cast_xXloop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_cast_x_0"è
qloop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_rngreadandskip_pfor_strided_slicesloop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_rngreadandskip_pfor_strided_slice_0"Ô
gloop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_rngreadandskip_resourceiloop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_rngreadandskip_resource_0*(
_construction_contextkEagerRuntime*#
_input_shapes
: : : : : : : : 2
Cloop_body/stateful_uniform/RngReadAndSkip/pfor/while/RngReadAndSkipCloop_body/stateful_uniform/RngReadAndSkip/pfor/while/RngReadAndSkip: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 

c
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_5278

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ç
F
*__inference_random_flip_layer_call_fn_7797

inputs
identityº
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_random_flip_layer_call_and_return_conditional_losses_5304j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ù
@__inference_conv2d_layer_call_and_return_conditional_losses_8827

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿüü*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿüüZ
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿüük
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿüüw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ç


random_flip_map_while_cond_6792<
8random_flip_map_while_random_flip_map_while_loop_counter7
3random_flip_map_while_random_flip_map_strided_slice%
!random_flip_map_while_placeholder'
#random_flip_map_while_placeholder_1<
8random_flip_map_while_less_random_flip_map_strided_sliceR
Nrandom_flip_map_while_random_flip_map_while_cond_6792___redundant_placeholder0"
random_flip_map_while_identity
 
random_flip/map/while/LessLess!random_flip_map_while_placeholder8random_flip_map_while_less_random_flip_map_strided_slice*
T0*
_output_shapes
: ´
random_flip/map/while/Less_1Less8random_flip_map_while_random_flip_map_while_loop_counter3random_flip_map_while_random_flip_map_strided_slice*
T0*
_output_shapes
: 
 random_flip/map/while/LogicalAnd
LogicalAnd random_flip/map/while/Less_1:z:0random_flip/map/while/Less:z:0*
_output_shapes
: q
random_flip/map/while/IdentityIdentity$random_flip/map/while/LogicalAnd:z:0*
T0
*
_output_shapes
: "I
random_flip_map_while_identity'random_flip/map/while/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : : : :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
:
Ç
_
C__inference_rescaling_layer_call_and_return_conditional_losses_5321

inputs
identityK
Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *<S
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿQ
Cast_1CastCast_1/x:output:0*

DstT0*

SrcT0*
_output_shapes
: _
mulMulinputsCast/x:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
addAddV2mul:z:0
Cast_1:y:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentityadd:z:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
­

Hloop_body_stateful_uniform_StatelessRandomUniformV2_pfor_while_cond_5853
loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_counter
loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_maximum_iterationsN
Jloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_placeholderP
Lloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_placeholder_1
loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_less_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_strided_slice¥
 loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_cond_5853___redundant_placeholder0¥
 loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_cond_5853___redundant_placeholder1¥
 loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_cond_5853___redundant_placeholder2¥
 loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_cond_5853___redundant_placeholder3K
Gloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_identity
Å
Cloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/LessLessJloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_placeholderloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_less_loop_body_stateful_uniform_statelessrandomuniformv2_pfor_strided_slice*
T0*
_output_shapes
: ½
Gloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/IdentityIdentityGloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/Less:z:0*
T0
*
_output_shapes
: "
Gloop_body_stateful_uniform_statelessrandomuniformv2_pfor_while_identityPloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
: : : : : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
:
Ç
F
*__inference_random_flip_layer_call_fn_7802

inputs
identityº
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_random_flip_layer_call_and_return_conditional_losses_6495j
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ñ
Í

Nrandom_rotation_loop_body_stateful_uniform_RngReadAndSkip_pfor_while_cond_6988
random_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_random_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_loop_counter¡
random_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_random_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_maximum_iterationsT
Prandom_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_placeholderV
Rrandom_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_placeholder_1
random_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_less_random_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_strided_slice±
¬random_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_random_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_cond_6988___redundant_placeholder0±
¬random_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_random_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_cond_6988___redundant_placeholder1±
¬random_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_random_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_cond_6988___redundant_placeholder2Q
Mrandom_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_identity
Ý
Irandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/while/LessLessPrandom_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_placeholderrandom_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_less_random_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_strided_slice*
T0*
_output_shapes
: É
Mrandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/while/IdentityIdentityMrandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/while/Less:z:0*
T0
*
_output_shapes
: "§
Mrandom_rotation_loop_body_stateful_uniform_rngreadandskip_pfor_while_identityVrandom_rotation/loop_body/stateful_uniform/RngReadAndSkip/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*)
_input_shapes
: : : : : :::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
:
«
»

Cloop_body_transform_ImageProjectiveTransformV3_pfor_while_cond_6354
loop_body_transform_imageprojectivetransformv3_pfor_while_loop_body_transform_imageprojectivetransformv3_pfor_while_loop_counter
loop_body_transform_imageprojectivetransformv3_pfor_while_loop_body_transform_imageprojectivetransformv3_pfor_while_maximum_iterationsI
Eloop_body_transform_imageprojectivetransformv3_pfor_while_placeholderK
Gloop_body_transform_imageprojectivetransformv3_pfor_while_placeholder_1
loop_body_transform_imageprojectivetransformv3_pfor_while_less_loop_body_transform_imageprojectivetransformv3_pfor_strided_slice
loop_body_transform_imageprojectivetransformv3_pfor_while_loop_body_transform_imageprojectivetransformv3_pfor_while_cond_6354___redundant_placeholder0
loop_body_transform_imageprojectivetransformv3_pfor_while_loop_body_transform_imageprojectivetransformv3_pfor_while_cond_6354___redundant_placeholder1
loop_body_transform_imageprojectivetransformv3_pfor_while_loop_body_transform_imageprojectivetransformv3_pfor_while_cond_6354___redundant_placeholder2
loop_body_transform_imageprojectivetransformv3_pfor_while_loop_body_transform_imageprojectivetransformv3_pfor_while_cond_6354___redundant_placeholder3F
Bloop_body_transform_imageprojectivetransformv3_pfor_while_identity
±
>loop_body/transform/ImageProjectiveTransformV3/pfor/while/LessLessEloop_body_transform_imageprojectivetransformv3_pfor_while_placeholderloop_body_transform_imageprojectivetransformv3_pfor_while_less_loop_body_transform_imageprojectivetransformv3_pfor_strided_slice*
T0*
_output_shapes
: ³
Bloop_body/transform/ImageProjectiveTransformV3/pfor/while/IdentityIdentityBloop_body/transform/ImageProjectiveTransformV3/pfor/while/Less:z:0*
T0
*
_output_shapes
: "
Bloop_body_transform_imageprojectivetransformv3_pfor_while_identityKloop_body/transform/ImageProjectiveTransformV3/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
: : : : : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
::

_output_shapes
::

_output_shapes
:


ë
)__inference_sequential_layer_call_fn_6735

inputs
unknown:	#
	unknown_0:
	unknown_1:#
	unknown_2:0
	unknown_3:0#
	unknown_4:0`
	unknown_5:`
	unknown_6:
à
	unknown_7:
identity¢StatefulPartitionedCall³
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

	*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_6554o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¢4
ü
>loop_body_stateful_uniform_RngReadAndSkip_pfor_while_body_8035z
vloop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_rngreadandskip_pfor_while_loop_counter
|loop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_rngreadandskip_pfor_while_maximum_iterationsD
@loop_body_stateful_uniform_rngreadandskip_pfor_while_placeholderF
Bloop_body_stateful_uniform_rngreadandskip_pfor_while_placeholder_1w
sloop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_rngreadandskip_pfor_strided_slice_0w
iloop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_rngreadandskip_resource_0:	\
Xloop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_cast_x_0\
Xloop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_cast_1_0A
=loop_body_stateful_uniform_rngreadandskip_pfor_while_identityC
?loop_body_stateful_uniform_rngreadandskip_pfor_while_identity_1C
?loop_body_stateful_uniform_rngreadandskip_pfor_while_identity_2C
?loop_body_stateful_uniform_rngreadandskip_pfor_while_identity_3u
qloop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_rngreadandskip_pfor_strided_sliceu
gloop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_rngreadandskip_resource:	Z
Vloop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_cast_xZ
Vloop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_cast_1¢Cloop_body/stateful_uniform/RngReadAndSkip/pfor/while/RngReadAndSkip
Cloop_body/stateful_uniform/RngReadAndSkip/pfor/while/RngReadAndSkipRngReadAndSkipiloop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_rngreadandskip_resource_0Xloop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_cast_x_0Xloop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_cast_1_0*
_output_shapes
:
Cloop_body/stateful_uniform/RngReadAndSkip/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
?loop_body/stateful_uniform/RngReadAndSkip/pfor/while/ExpandDims
ExpandDimsKloop_body/stateful_uniform/RngReadAndSkip/pfor/while/RngReadAndSkip:value:0Lloop_body/stateful_uniform/RngReadAndSkip/pfor/while/ExpandDims/dim:output:0*
T0	*
_output_shapes

:þ
Yloop_body/stateful_uniform/RngReadAndSkip/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemBloop_body_stateful_uniform_rngreadandskip_pfor_while_placeholder_1@loop_body_stateful_uniform_rngreadandskip_pfor_while_placeholderHloop_body/stateful_uniform/RngReadAndSkip/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0	:éèÐ|
:loop_body/stateful_uniform/RngReadAndSkip/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :é
8loop_body/stateful_uniform/RngReadAndSkip/pfor/while/addAddV2@loop_body_stateful_uniform_rngreadandskip_pfor_while_placeholderCloop_body/stateful_uniform/RngReadAndSkip/pfor/while/add/y:output:0*
T0*
_output_shapes
: ~
<loop_body/stateful_uniform/RngReadAndSkip/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :£
:loop_body/stateful_uniform/RngReadAndSkip/pfor/while/add_1AddV2vloop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_rngreadandskip_pfor_while_loop_counterEloop_body/stateful_uniform/RngReadAndSkip/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: æ
=loop_body/stateful_uniform/RngReadAndSkip/pfor/while/IdentityIdentity>loop_body/stateful_uniform/RngReadAndSkip/pfor/while/add_1:z:0:^loop_body/stateful_uniform/RngReadAndSkip/pfor/while/NoOp*
T0*
_output_shapes
: ¦
?loop_body/stateful_uniform/RngReadAndSkip/pfor/while/Identity_1Identity|loop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_rngreadandskip_pfor_while_maximum_iterations:^loop_body/stateful_uniform/RngReadAndSkip/pfor/while/NoOp*
T0*
_output_shapes
: æ
?loop_body/stateful_uniform/RngReadAndSkip/pfor/while/Identity_2Identity<loop_body/stateful_uniform/RngReadAndSkip/pfor/while/add:z:0:^loop_body/stateful_uniform/RngReadAndSkip/pfor/while/NoOp*
T0*
_output_shapes
: 
?loop_body/stateful_uniform/RngReadAndSkip/pfor/while/Identity_3Identityiloop_body/stateful_uniform/RngReadAndSkip/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0:^loop_body/stateful_uniform/RngReadAndSkip/pfor/while/NoOp*
T0*
_output_shapes
: Á
9loop_body/stateful_uniform/RngReadAndSkip/pfor/while/NoOpNoOpD^loop_body/stateful_uniform/RngReadAndSkip/pfor/while/RngReadAndSkip*"
_acd_function_control_output(*
_output_shapes
 "
=loop_body_stateful_uniform_rngreadandskip_pfor_while_identityFloop_body/stateful_uniform/RngReadAndSkip/pfor/while/Identity:output:0"
?loop_body_stateful_uniform_rngreadandskip_pfor_while_identity_1Hloop_body/stateful_uniform/RngReadAndSkip/pfor/while/Identity_1:output:0"
?loop_body_stateful_uniform_rngreadandskip_pfor_while_identity_2Hloop_body/stateful_uniform/RngReadAndSkip/pfor/while/Identity_2:output:0"
?loop_body_stateful_uniform_rngreadandskip_pfor_while_identity_3Hloop_body/stateful_uniform/RngReadAndSkip/pfor/while/Identity_3:output:0"²
Vloop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_cast_1Xloop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_cast_1_0"²
Vloop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_cast_xXloop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_cast_x_0"è
qloop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_rngreadandskip_pfor_strided_slicesloop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_rngreadandskip_pfor_strided_slice_0"Ô
gloop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_rngreadandskip_resourceiloop_body_stateful_uniform_rngreadandskip_pfor_while_loop_body_stateful_uniform_rngreadandskip_resource_0*(
_construction_contextkEagerRuntime*#
_input_shapes
: : : : : : : : 2
Cloop_body/stateful_uniform/RngReadAndSkip/pfor/while/RngReadAndSkipCloop_body/stateful_uniform/RngReadAndSkip/pfor/while/RngReadAndSkip: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
«-
â
D__inference_sequential_layer_call_and_return_conditional_losses_6554

inputs"
random_rotation_6525:	%
conv2d_6529:
conv2d_6531:'
conv2d_1_6535:0
conv2d_1_6537:0'
conv2d_2_6541:0`
conv2d_2_6543:`

dense_6548:
à

dense_6550:
identity¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCall¢ conv2d_2/StatefulPartitionedCall¢dense/StatefulPartitionedCall¢dropout/StatefulPartitionedCall¢'random_rotation/StatefulPartitionedCallÆ
random_flip/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_random_flip_layer_call_and_return_conditional_losses_6495
'random_rotation/StatefulPartitionedCallStatefulPartitionedCall$random_flip/PartitionedCall:output:0random_rotation_6525*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_random_rotation_layer_call_and_return_conditional_losses_6423ì
rescaling/PartitionedCallPartitionedCall0random_rotation/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_rescaling_layer_call_and_return_conditional_losses_5321
conv2d/StatefulPartitionedCallStatefulPartitionedCall"rescaling/PartitionedCall:output:0conv2d_6529conv2d_6531*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿüü*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_5334é
max_pooling2d/PartitionedCallPartitionedCall'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~~* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_5278
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall&max_pooling2d/PartitionedCall:output:0conv2d_1_6535conv2d_1_6537*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿzz0*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_1_layer_call_and_return_conditional_losses_5352ï
max_pooling2d_1/PartitionedCallPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ==0* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_5290
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall(max_pooling2d_1/PartitionedCall:output:0conv2d_2_6541conv2d_2_6543*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ99`*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *K
fFRD
B__inference_conv2d_2_layer_call_and_return_conditional_losses_5370Ù
flatten/PartitionedCallPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿà* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_5382à
dropout/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:ÿÿÿÿÿÿÿÿÿà* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dropout_layer_call_and_return_conditional_losses_5458
dense/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0
dense_6548
dense_6550*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *H
fCRA
?__inference_dense_layer_call_and_return_conditional_losses_5402u
IdentityIdentity&dense/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall(^random_rotation/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:ÿÿÿÿÿÿÿÿÿ: : : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2R
'random_rotation/StatefulPartitionedCall'random_rotation/StatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


Ý
)__inference_sequential_layer_call_fn_5428
random_flip_input!
unknown:
	unknown_0:#
	unknown_1:0
	unknown_2:0#
	unknown_3:0`
	unknown_4:`
	unknown_5:
à
	unknown_6:
identity¢StatefulPartitionedCall²
StatefulPartitionedCallStatefulPartitionedCallrandom_flip_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_5409o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:ÿÿÿÿÿÿÿÿÿ: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:d `
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+
_user_specified_namerandom_flip_input
Ù=

Grandom_rotation_loop_body_stateful_uniform_Bitcast_pfor_while_body_7054
random_rotation_loop_body_stateful_uniform_bitcast_pfor_while_random_rotation_loop_body_stateful_uniform_bitcast_pfor_while_loop_counter
random_rotation_loop_body_stateful_uniform_bitcast_pfor_while_random_rotation_loop_body_stateful_uniform_bitcast_pfor_while_maximum_iterationsM
Irandom_rotation_loop_body_stateful_uniform_bitcast_pfor_while_placeholderO
Krandom_rotation_loop_body_stateful_uniform_bitcast_pfor_while_placeholder_1
random_rotation_loop_body_stateful_uniform_bitcast_pfor_while_random_rotation_loop_body_stateful_uniform_bitcast_pfor_strided_slice_0
random_rotation_loop_body_stateful_uniform_bitcast_pfor_while_strided_slice_random_rotation_loop_body_stateful_uniform_strided_slice_pfor_stridedslice_0	J
Frandom_rotation_loop_body_stateful_uniform_bitcast_pfor_while_identityL
Hrandom_rotation_loop_body_stateful_uniform_bitcast_pfor_while_identity_1L
Hrandom_rotation_loop_body_stateful_uniform_bitcast_pfor_while_identity_2L
Hrandom_rotation_loop_body_stateful_uniform_bitcast_pfor_while_identity_3
random_rotation_loop_body_stateful_uniform_bitcast_pfor_while_random_rotation_loop_body_stateful_uniform_bitcast_pfor_strided_slice
random_rotation_loop_body_stateful_uniform_bitcast_pfor_while_strided_slice_random_rotation_loop_body_stateful_uniform_strided_slice_pfor_stridedslice	
Crandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
Arandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/while/addAddV2Irandom_rotation_loop_body_stateful_uniform_bitcast_pfor_while_placeholderLrandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/while/add/y:output:0*
T0*
_output_shapes
: 
Srandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : °
Qrandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/while/strided_slice/stackPackIrandom_rotation_loop_body_stateful_uniform_bitcast_pfor_while_placeholder\random_rotation/loop_body/stateful_uniform/Bitcast/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
:
Urandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : °
Srandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/while/strided_slice/stack_1PackErandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/while/add:z:0^random_rotation/loop_body/stateful_uniform/Bitcast/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
:¤
Srandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      í
Krandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/while/strided_sliceStridedSlicerandom_rotation_loop_body_stateful_uniform_bitcast_pfor_while_strided_slice_random_rotation_loop_body_stateful_uniform_strided_slice_pfor_stridedslice_0Zrandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/while/strided_slice/stack:output:0\random_rotation/loop_body/stateful_uniform/Bitcast/pfor/while/strided_slice/stack_1:output:0\random_rotation/loop_body/stateful_uniform/Bitcast/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*
ellipsis_mask*
shrink_axis_mask×
Erandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/while/BitcastBitcastTrandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/while/strided_slice:output:0*
T0	*
_output_shapes
:*

type0
Lrandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ¦
Hrandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/while/ExpandDims
ExpandDimsNrandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/while/Bitcast:output:0Urandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/while/ExpandDims/dim:output:0*
T0*
_output_shapes

:¢
brandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemKrandom_rotation_loop_body_stateful_uniform_bitcast_pfor_while_placeholder_1Irandom_rotation_loop_body_stateful_uniform_bitcast_pfor_while_placeholderQrandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0:éèÌ
Erandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
Crandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/while/add_1AddV2Irandom_rotation_loop_body_stateful_uniform_bitcast_pfor_while_placeholderNrandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: 
Erandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :È
Crandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/while/add_2AddV2random_rotation_loop_body_stateful_uniform_bitcast_pfor_while_random_rotation_loop_body_stateful_uniform_bitcast_pfor_while_loop_counterNrandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/while/add_2/y:output:0*
T0*
_output_shapes
: ¼
Frandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/while/IdentityIdentityGrandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/while/add_2:z:0*
T0*
_output_shapes
: 
Hrandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/while/Identity_1Identityrandom_rotation_loop_body_stateful_uniform_bitcast_pfor_while_random_rotation_loop_body_stateful_uniform_bitcast_pfor_while_maximum_iterations*
T0*
_output_shapes
: ¾
Hrandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/while/Identity_2IdentityGrandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/while/add_1:z:0*
T0*
_output_shapes
: é
Hrandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/while/Identity_3Identityrrandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "
Frandom_rotation_loop_body_stateful_uniform_bitcast_pfor_while_identityOrandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/while/Identity:output:0"
Hrandom_rotation_loop_body_stateful_uniform_bitcast_pfor_while_identity_1Qrandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/while/Identity_1:output:0"
Hrandom_rotation_loop_body_stateful_uniform_bitcast_pfor_while_identity_2Qrandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/while/Identity_2:output:0"
Hrandom_rotation_loop_body_stateful_uniform_bitcast_pfor_while_identity_3Qrandom_rotation/loop_body/stateful_uniform/Bitcast/pfor/while/Identity_3:output:0"
random_rotation_loop_body_stateful_uniform_bitcast_pfor_while_random_rotation_loop_body_stateful_uniform_bitcast_pfor_strided_slicerandom_rotation_loop_body_stateful_uniform_bitcast_pfor_while_random_rotation_loop_body_stateful_uniform_bitcast_pfor_strided_slice_0"´
random_rotation_loop_body_stateful_uniform_bitcast_pfor_while_strided_slice_random_rotation_loop_body_stateful_uniform_strided_slice_pfor_stridedslicerandom_rotation_loop_body_stateful_uniform_bitcast_pfor_while_strided_slice_random_rotation_loop_body_stateful_uniform_strided_slice_pfor_stridedslice_0*(
_construction_contextkEagerRuntime*0
_input_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ÂX

Cloop_body_transform_ImageProjectiveTransformV3_pfor_while_body_6355
loop_body_transform_imageprojectivetransformv3_pfor_while_loop_body_transform_imageprojectivetransformv3_pfor_while_loop_counter
loop_body_transform_imageprojectivetransformv3_pfor_while_loop_body_transform_imageprojectivetransformv3_pfor_while_maximum_iterationsI
Eloop_body_transform_imageprojectivetransformv3_pfor_while_placeholderK
Gloop_body_transform_imageprojectivetransformv3_pfor_while_placeholder_1
}loop_body_transform_imageprojectivetransformv3_pfor_while_loop_body_transform_imageprojectivetransformv3_pfor_strided_slice_0r
nloop_body_transform_imageprojectivetransformv3_pfor_while_strided_slice_loop_body_expanddims_pfor_expanddims_0~
zloop_body_transform_imageprojectivetransformv3_pfor_while_strided_slice_1_loop_body_rotation_matrix_concat_pfor_concat_1_0a
]loop_body_transform_imageprojectivetransformv3_pfor_while_loop_body_transform_strided_slice_0^
Zloop_body_transform_imageprojectivetransformv3_pfor_while_loop_body_transform_fill_value_0F
Bloop_body_transform_imageprojectivetransformv3_pfor_while_identityH
Dloop_body_transform_imageprojectivetransformv3_pfor_while_identity_1H
Dloop_body_transform_imageprojectivetransformv3_pfor_while_identity_2H
Dloop_body_transform_imageprojectivetransformv3_pfor_while_identity_3
{loop_body_transform_imageprojectivetransformv3_pfor_while_loop_body_transform_imageprojectivetransformv3_pfor_strided_slicep
lloop_body_transform_imageprojectivetransformv3_pfor_while_strided_slice_loop_body_expanddims_pfor_expanddims|
xloop_body_transform_imageprojectivetransformv3_pfor_while_strided_slice_1_loop_body_rotation_matrix_concat_pfor_concat_1_
[loop_body_transform_imageprojectivetransformv3_pfor_while_loop_body_transform_strided_slice\
Xloop_body_transform_imageprojectivetransformv3_pfor_while_loop_body_transform_fill_value
?loop_body/transform/ImageProjectiveTransformV3/pfor/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :ø
=loop_body/transform/ImageProjectiveTransformV3/pfor/while/addAddV2Eloop_body_transform_imageprojectivetransformv3_pfor_while_placeholderHloop_body/transform/ImageProjectiveTransformV3/pfor/while/add/y:output:0*
T0*
_output_shapes
: 
Oloop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ¤
Mloop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice/stackPackEloop_body_transform_imageprojectivetransformv3_pfor_while_placeholderXloop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice/stack/1:output:0*
N*
T0*
_output_shapes
:
Qloop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ¤
Oloop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice/stack_1PackAloop_body/transform/ImageProjectiveTransformV3/pfor/while/add:z:0Zloop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice/stack_1/1:output:0*
N*
T0*
_output_shapes
: 
Oloop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      À
Gloop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_sliceStridedSlicenloop_body_transform_imageprojectivetransformv3_pfor_while_strided_slice_loop_body_expanddims_pfor_expanddims_0Vloop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice/stack:output:0Xloop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice/stack_1:output:0Xloop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:*
ellipsis_mask*
shrink_axis_mask
Aloop_body/transform/ImageProjectiveTransformV3/pfor/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :ü
?loop_body/transform/ImageProjectiveTransformV3/pfor/while/add_1AddV2Eloop_body_transform_imageprojectivetransformv3_pfor_while_placeholderJloop_body/transform/ImageProjectiveTransformV3/pfor/while/add_1/y:output:0*
T0*
_output_shapes
: 
Qloop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B : ¨
Oloop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice_1/stackPackEloop_body_transform_imageprojectivetransformv3_pfor_while_placeholderZloop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice_1/stack/1:output:0*
N*
T0*
_output_shapes
:
Sloop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice_1/stack_1/1Const*
_output_shapes
: *
dtype0*
value	B : ª
Qloop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice_1/stack_1PackCloop_body/transform/ImageProjectiveTransformV3/pfor/while/add_1:z:0\loop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice_1/stack_1/1:output:0*
N*
T0*
_output_shapes
:¢
Qloop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ê
Iloop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice_1StridedSlicezloop_body_transform_imageprojectivetransformv3_pfor_while_strided_slice_1_loop_body_rotation_matrix_concat_pfor_concat_1_0Xloop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice_1/stack:output:0Zloop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice_1/stack_1:output:0Zloop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*
ellipsis_mask*
shrink_axis_maskÏ
Tloop_body/transform/ImageProjectiveTransformV3/pfor/while/ImageProjectiveTransformV3ImageProjectiveTransformV3Ploop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice:output:0Rloop_body/transform/ImageProjectiveTransformV3/pfor/while/strided_slice_1:output:0]loop_body_transform_imageprojectivetransformv3_pfor_while_loop_body_transform_strided_slice_0Zloop_body_transform_imageprojectivetransformv3_pfor_while_loop_body_transform_fill_value_0*8
_output_shapes&
$:"ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR
Hloop_body/transform/ImageProjectiveTransformV3/pfor/while/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : ×
Dloop_body/transform/ImageProjectiveTransformV3/pfor/while/ExpandDims
ExpandDimsiloop_body/transform/ImageProjectiveTransformV3/pfor/while/ImageProjectiveTransformV3:transformed_images:0Qloop_body/transform/ImageProjectiveTransformV3/pfor/while/ExpandDims/dim:output:0*
T0*<
_output_shapes*
(:&ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
^loop_body/transform/ImageProjectiveTransformV3/pfor/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemGloop_body_transform_imageprojectivetransformv3_pfor_while_placeholder_1Eloop_body_transform_imageprojectivetransformv3_pfor_while_placeholderMloop_body/transform/ImageProjectiveTransformV3/pfor/while/ExpandDims:output:0*
_output_shapes
: *
element_dtype0:éèÒ
Aloop_body/transform/ImageProjectiveTransformV3/pfor/while/add_2/yConst*
_output_shapes
: *
dtype0*
value	B :ü
?loop_body/transform/ImageProjectiveTransformV3/pfor/while/add_2AddV2Eloop_body_transform_imageprojectivetransformv3_pfor_while_placeholderJloop_body/transform/ImageProjectiveTransformV3/pfor/while/add_2/y:output:0*
T0*
_output_shapes
: 
Aloop_body/transform/ImageProjectiveTransformV3/pfor/while/add_3/yConst*
_output_shapes
: *
dtype0*
value	B :¸
?loop_body/transform/ImageProjectiveTransformV3/pfor/while/add_3AddV2loop_body_transform_imageprojectivetransformv3_pfor_while_loop_body_transform_imageprojectivetransformv3_pfor_while_loop_counterJloop_body/transform/ImageProjectiveTransformV3/pfor/while/add_3/y:output:0*
T0*
_output_shapes
: ´
Bloop_body/transform/ImageProjectiveTransformV3/pfor/while/IdentityIdentityCloop_body/transform/ImageProjectiveTransformV3/pfor/while/add_3:z:0*
T0*
_output_shapes
: ú
Dloop_body/transform/ImageProjectiveTransformV3/pfor/while/Identity_1Identityloop_body_transform_imageprojectivetransformv3_pfor_while_loop_body_transform_imageprojectivetransformv3_pfor_while_maximum_iterations*
T0*
_output_shapes
: ¶
Dloop_body/transform/ImageProjectiveTransformV3/pfor/while/Identity_2IdentityCloop_body/transform/ImageProjectiveTransformV3/pfor/while/add_2:z:0*
T0*
_output_shapes
: á
Dloop_body/transform/ImageProjectiveTransformV3/pfor/while/Identity_3Identitynloop_body/transform/ImageProjectiveTransformV3/pfor/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "
Bloop_body_transform_imageprojectivetransformv3_pfor_while_identityKloop_body/transform/ImageProjectiveTransformV3/pfor/while/Identity:output:0"
Dloop_body_transform_imageprojectivetransformv3_pfor_while_identity_1Mloop_body/transform/ImageProjectiveTransformV3/pfor/while/Identity_1:output:0"
Dloop_body_transform_imageprojectivetransformv3_pfor_while_identity_2Mloop_body/transform/ImageProjectiveTransformV3/pfor/while/Identity_2:output:0"
Dloop_body_transform_imageprojectivetransformv3_pfor_while_identity_3Mloop_body/transform/ImageProjectiveTransformV3/pfor/while/Identity_3:output:0"¶
Xloop_body_transform_imageprojectivetransformv3_pfor_while_loop_body_transform_fill_valueZloop_body_transform_imageprojectivetransformv3_pfor_while_loop_body_transform_fill_value_0"ü
{loop_body_transform_imageprojectivetransformv3_pfor_while_loop_body_transform_imageprojectivetransformv3_pfor_strided_slice}loop_body_transform_imageprojectivetransformv3_pfor_while_loop_body_transform_imageprojectivetransformv3_pfor_strided_slice_0"¼
[loop_body_transform_imageprojectivetransformv3_pfor_while_loop_body_transform_strided_slice]loop_body_transform_imageprojectivetransformv3_pfor_while_loop_body_transform_strided_slice_0"ö
xloop_body_transform_imageprojectivetransformv3_pfor_while_strided_slice_1_loop_body_rotation_matrix_concat_pfor_concat_1zloop_body_transform_imageprojectivetransformv3_pfor_while_strided_slice_1_loop_body_rotation_matrix_concat_pfor_concat_1_0"Þ
lloop_body_transform_imageprojectivetransformv3_pfor_while_strided_slice_loop_body_expanddims_pfor_expanddimsnloop_body_transform_imageprojectivetransformv3_pfor_while_strided_slice_loop_body_expanddims_pfor_expanddims_0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :;7
5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿ:1-
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ: 

_output_shapes
::

_output_shapes
: 

û
B__inference_conv2d_2_layer_call_and_return_conditional_losses_8887

inputs8
conv2d_readvariableop_resource:0`-
biasadd_readvariableop_resource:`
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:0`*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ99`*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:`*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ99`X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ99`i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ99`w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ==0: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ==0
 
_user_specified_nameinputs
ê

%__inference_conv2d_layer_call_fn_8816

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallß
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:ÿÿÿÿÿÿÿÿÿüü*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_conv2d_layer_call_and_return_conditional_losses_5334y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿüü`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ä$
æ
random_flip_map_while_body_6793<
8random_flip_map_while_random_flip_map_while_loop_counter7
3random_flip_map_while_random_flip_map_strided_slice%
!random_flip_map_while_placeholder'
#random_flip_map_while_placeholder_1;
7random_flip_map_while_random_flip_map_strided_slice_1_0w
srandom_flip_map_while_tensorarrayv2read_tensorlistgetitem_random_flip_map_tensorarrayunstack_tensorlistfromtensor_0"
random_flip_map_while_identity$
 random_flip_map_while_identity_1$
 random_flip_map_while_identity_2$
 random_flip_map_while_identity_39
5random_flip_map_while_random_flip_map_strided_slice_1u
qrandom_flip_map_while_tensorarrayv2read_tensorlistgetitem_random_flip_map_tensorarrayunstack_tensorlistfromtensor
Grandom_flip/map/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*!
valueB"         ó
9random_flip/map/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsrandom_flip_map_while_tensorarrayv2read_tensorlistgetitem_random_flip_map_tensorarrayunstack_tensorlistfromtensor_0!random_flip_map_while_placeholderPrandom_flip/map/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*$
_output_shapes
:*
element_dtype0
8random_flip/map/while/flip_left_right/control_dependencyIdentity@random_flip/map/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*L
_classB
@>loc:@random_flip/map/while/TensorArrayV2Read/TensorListGetItem*$
_output_shapes
:~
4random_flip/map/while/flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:í
/random_flip/map/while/flip_left_right/ReverseV2	ReverseV2Arandom_flip/map/while/flip_left_right/control_dependency:output:0=random_flip/map/while/flip_left_right/ReverseV2/axis:output:0*
T0*$
_output_shapes
:î
5random_flip/map/while/flip_up_down/control_dependencyIdentity8random_flip/map/while/flip_left_right/ReverseV2:output:0*
T0*B
_class8
64loc:@random_flip/map/while/flip_left_right/ReverseV2*$
_output_shapes
:{
1random_flip/map/while/flip_up_down/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB: ä
,random_flip/map/while/flip_up_down/ReverseV2	ReverseV2>random_flip/map/while/flip_up_down/control_dependency:output:0:random_flip/map/while/flip_up_down/ReverseV2/axis:output:0*
T0*$
_output_shapes
:
:random_flip/map/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem#random_flip_map_while_placeholder_1!random_flip_map_while_placeholder5random_flip/map/while/flip_up_down/ReverseV2:output:0*
_output_shapes
: *
element_dtype0:éèÒ]
random_flip/map/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
random_flip/map/while/addAddV2!random_flip_map_while_placeholder$random_flip/map/while/add/y:output:0*
T0*
_output_shapes
: _
random_flip/map/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :§
random_flip/map/while/add_1AddV28random_flip_map_while_random_flip_map_while_loop_counter&random_flip/map/while/add_1/y:output:0*
T0*
_output_shapes
: l
random_flip/map/while/IdentityIdentityrandom_flip/map/while/add_1:z:0*
T0*
_output_shapes
: 
 random_flip/map/while/Identity_1Identity3random_flip_map_while_random_flip_map_strided_slice*
T0*
_output_shapes
: l
 random_flip/map/while/Identity_2Identityrandom_flip/map/while/add:z:0*
T0*
_output_shapes
: 
 random_flip/map/while/Identity_3IdentityJrandom_flip/map/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: "I
random_flip_map_while_identity'random_flip/map/while/Identity:output:0"M
 random_flip_map_while_identity_1)random_flip/map/while/Identity_1:output:0"M
 random_flip_map_while_identity_2)random_flip/map/while/Identity_2:output:0"M
 random_flip_map_while_identity_3)random_flip/map/while/Identity_3:output:0"p
5random_flip_map_while_random_flip_map_strided_slice_17random_flip_map_while_random_flip_map_strided_slice_1_0"è
qrandom_flip_map_while_tensorarrayv2read_tensorlistgetitem_random_flip_map_tensorarrayunstack_tensorlistfromtensorsrandom_flip_map_while_tensorarrayv2read_tensorlistgetitem_random_flip_map_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
¢J

__inference__traced_save_9083
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop3
/savev2_random_flip_statevar_read_readvariableop	7
3savev2_random_rotation_statevar_read_readvariableop	&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableop5
1savev2_adam_conv2d_2_kernel_m_read_readvariableop3
/savev2_adam_conv2d_2_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableop5
1savev2_adam_conv2d_2_kernel_v_read_readvariableop3
/savev2_adam_conv2d_2_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Ç
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*ð
valueæBã$B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEBJlayer-0/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEBJlayer-1/_random_generator/_generator/_state_var/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHµ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:$*
dtype0*[
valueRBP$B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ë
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop/savev2_random_flip_statevar_read_readvariableop3savev2_random_rotation_statevar_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop1savev2_adam_conv2d_2_kernel_m_read_readvariableop/savev2_adam_conv2d_2_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop1savev2_adam_conv2d_2_kernel_v_read_readvariableop/savev2_adam_conv2d_2_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *2
dtypes(
&2$			
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*Å
_input_shapes³
°: :::0:0:0`:`:
à:: : : : : ::: : : : :::0:0:0`:`:
à::::0:0:0`:`:
à:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:0: 

_output_shapes
:0:,(
&
_output_shapes
:0`: 

_output_shapes
:`:&"
 
_output_shapes
:
à: 

_output_shapes
::	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:0: 

_output_shapes
:0:,(
&
_output_shapes
:0`: 

_output_shapes
:`:&"
 
_output_shapes
:
à: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:0: 

_output_shapes
:0:, (
&
_output_shapes
:0`: !

_output_shapes
:`:&""
 
_output_shapes
:
à: #

_output_shapes
::$

_output_shapes
: 
±¿

I__inference_random_rotation_layer_call_and_return_conditional_losses_6423

inputs@
2loop_body_stateful_uniform_rngreadandskip_resource:	
identity¢)loop_body/stateful_uniform/RngReadAndSkip¢4loop_body/stateful_uniform/RngReadAndSkip/pfor/while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
Rank/packedPackstrided_slice:output:0*
N*
T0*
_output_shapes
:F
RankConst*
_output_shapes
: *
dtype0*
value	B :M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :e
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
:W
	Max/inputPackstrided_slice:output:0*
N*
T0*
_output_shapes
:O
MaxMaxMax/input:output:0range:output:0*
T0*
_output_shapes
: h
&loop_body/PlaceholderWithDefault/inputConst*
_output_shapes
: *
dtype0*
value	B : 
 loop_body/PlaceholderWithDefaultPlaceholderWithDefault/loop_body/PlaceholderWithDefault/input:output:0*
_output_shapes
: *
dtype0*
shape: E
loop_body/ShapeShapeinputs*
T0*
_output_shapes
:g
loop_body/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
loop_body/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
loop_body/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
loop_body/strided_sliceStridedSliceloop_body/Shape:output:0&loop_body/strided_slice/stack:output:0(loop_body/strided_slice/stack_1:output:0(loop_body/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
loop_body/Greater/yConst*
_output_shapes
: *
dtype0*
value	B :}
loop_body/GreaterGreater loop_body/strided_slice:output:0loop_body/Greater/y:output:0*
T0*
_output_shapes
: V
loop_body/SelectV2/eConst*
_output_shapes
: *
dtype0*
value	B :  
loop_body/SelectV2SelectV2loop_body/Greater:z:0)loop_body/PlaceholderWithDefault:output:0loop_body/SelectV2/e:output:0*
T0*
_output_shapes
: Y
loop_body/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ·
loop_body/GatherV2GatherV2inputsloop_body/SelectV2:output:0 loop_body/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*$
_output_shapes
:j
 loop_body/stateful_uniform/shapeConst*
_output_shapes
:*
dtype0*
valueB:c
loop_body/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|Ù ¿c
loop_body/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|Ù ?j
 loop_body/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
loop_body/stateful_uniform/ProdProd)loop_body/stateful_uniform/shape:output:0)loop_body/stateful_uniform/Const:output:0*
T0*
_output_shapes
: c
!loop_body/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :
!loop_body/stateful_uniform/Cast_1Cast(loop_body/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: Þ
)loop_body/stateful_uniform/RngReadAndSkipRngReadAndSkip2loop_body_stateful_uniform_rngreadandskip_resource*loop_body/stateful_uniform/Cast/x:output:0%loop_body/stateful_uniform/Cast_1:y:0*
_output_shapes
:x
.loop_body/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0loop_body/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0loop_body/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Þ
(loop_body/stateful_uniform/strided_sliceStridedSlice1loop_body/stateful_uniform/RngReadAndSkip:value:07loop_body/stateful_uniform/strided_slice/stack:output:09loop_body/stateful_uniform/strided_slice/stack_1:output:09loop_body/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask
"loop_body/stateful_uniform/BitcastBitcast1loop_body/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0z
0loop_body/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:|
2loop_body/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2loop_body/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ô
*loop_body/stateful_uniform/strided_slice_1StridedSlice1loop_body/stateful_uniform/RngReadAndSkip:value:09loop_body/stateful_uniform/strided_slice_1/stack:output:0;loop_body/stateful_uniform/strided_slice_1/stack_1:output:0;loop_body/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:
$loop_body/stateful_uniform/Bitcast_1Bitcast3loop_body/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0y
7loop_body/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :´
3loop_body/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2)loop_body/stateful_uniform/shape:output:0-loop_body/stateful_uniform/Bitcast_1:output:0+loop_body/stateful_uniform/Bitcast:output:0@loop_body/stateful_uniform/StatelessRandomUniformV2/alg:output:0*
_output_shapes
:
loop_body/stateful_uniform/subSub'loop_body/stateful_uniform/max:output:0'loop_body/stateful_uniform/min:output:0*
T0*
_output_shapes
: ¬
loop_body/stateful_uniform/mulMul<loop_body/stateful_uniform/StatelessRandomUniformV2:output:0"loop_body/stateful_uniform/sub:z:0*
T0*
_output_shapes
:
loop_body/stateful_uniformAddV2"loop_body/stateful_uniform/mul:z:0'loop_body/stateful_uniform/min:output:0*
T0*
_output_shapes
:Z
loop_body/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
loop_body/ExpandDims
ExpandDimsloop_body/GatherV2:output:0!loop_body/ExpandDims/dim:output:0*
T0*(
_output_shapes
:j
loop_body/Shape_1Const*
_output_shapes
:*
dtype0*%
valueB"            r
loop_body/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿt
!loop_body/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿk
!loop_body/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
loop_body/strided_slice_1StridedSliceloop_body/Shape_1:output:0(loop_body/strided_slice_1/stack:output:0*loop_body/strided_slice_1/stack_1:output:0*loop_body/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
loop_body/CastCast"loop_body/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: r
loop_body/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿt
!loop_body/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿk
!loop_body/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
loop_body/strided_slice_2StridedSliceloop_body/Shape_1:output:0(loop_body/strided_slice_2/stack:output:0*loop_body/strided_slice_2/stack_1:output:0*loop_body/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
loop_body/Cast_1Cast"loop_body/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: d
loop_body/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
loop_body/rotation_matrix/subSubloop_body/Cast_1:y:0(loop_body/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: i
loop_body/rotation_matrix/CosCosloop_body/stateful_uniform:z:0*
T0*
_output_shapes
:f
!loop_body/rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
loop_body/rotation_matrix/sub_1Subloop_body/Cast_1:y:0*loop_body/rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 
loop_body/rotation_matrix/mulMul!loop_body/rotation_matrix/Cos:y:0#loop_body/rotation_matrix/sub_1:z:0*
T0*
_output_shapes
:i
loop_body/rotation_matrix/SinSinloop_body/stateful_uniform:z:0*
T0*
_output_shapes
:f
!loop_body/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
loop_body/rotation_matrix/sub_2Subloop_body/Cast:y:0*loop_body/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 
loop_body/rotation_matrix/mul_1Mul!loop_body/rotation_matrix/Sin:y:0#loop_body/rotation_matrix/sub_2:z:0*
T0*
_output_shapes
:
loop_body/rotation_matrix/sub_3Sub!loop_body/rotation_matrix/mul:z:0#loop_body/rotation_matrix/mul_1:z:0*
T0*
_output_shapes
:
loop_body/rotation_matrix/sub_4Sub!loop_body/rotation_matrix/sub:z:0#loop_body/rotation_matrix/sub_3:z:0*
T0*
_output_shapes
:h
#loop_body/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @¤
!loop_body/rotation_matrix/truedivRealDiv#loop_body/rotation_matrix/sub_4:z:0,loop_body/rotation_matrix/truediv/y:output:0*
T0*
_output_shapes
:f
!loop_body/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
loop_body/rotation_matrix/sub_5Subloop_body/Cast:y:0*loop_body/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: k
loop_body/rotation_matrix/Sin_1Sinloop_body/stateful_uniform:z:0*
T0*
_output_shapes
:f
!loop_body/rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
loop_body/rotation_matrix/sub_6Subloop_body/Cast_1:y:0*loop_body/rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 
loop_body/rotation_matrix/mul_2Mul#loop_body/rotation_matrix/Sin_1:y:0#loop_body/rotation_matrix/sub_6:z:0*
T0*
_output_shapes
:k
loop_body/rotation_matrix/Cos_1Cosloop_body/stateful_uniform:z:0*
T0*
_output_shapes
:f
!loop_body/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
loop_body/rotation_matrix/sub_7Subloop_body/Cast:y:0*loop_body/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 
loop_body/rotation_matrix/mul_3Mul#loop_body/rotation_matrix/Cos_1:y:0#loop_body/rotation_matrix/sub_7:z:0*
T0*
_output_shapes
:
loop_body/rotation_matrix/addAddV2#loop_body/rotation_matrix/mul_2:z:0#loop_body/rotation_matrix/mul_3:z:0*
T0*
_output_shapes
:
loop_body/rotation_matrix/sub_8Sub#loop_body/rotation_matrix/sub_5:z:0!loop_body/rotation_matrix/add:z:0*
T0*
_output_shapes
:j
%loop_body/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @¨
#loop_body/rotation_matrix/truediv_1RealDiv#loop_body/rotation_matrix/sub_8:z:0.loop_body/rotation_matrix/truediv_1/y:output:0*
T0*
_output_shapes
:i
loop_body/rotation_matrix/ShapeConst*
_output_shapes
:*
dtype0*
valueB:w
-loop_body/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/loop_body/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/loop_body/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ó
'loop_body/rotation_matrix/strided_sliceStridedSlice(loop_body/rotation_matrix/Shape:output:06loop_body/rotation_matrix/strided_slice/stack:output:08loop_body/rotation_matrix/strided_slice/stack_1:output:08loop_body/rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
loop_body/rotation_matrix/Cos_2Cosloop_body/stateful_uniform:z:0*
T0*
_output_shapes
:
/loop_body/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        
1loop_body/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
1loop_body/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ý
)loop_body/rotation_matrix/strided_slice_1StridedSlice#loop_body/rotation_matrix/Cos_2:y:08loop_body/rotation_matrix/strided_slice_1/stack:output:0:loop_body/rotation_matrix/strided_slice_1/stack_1:output:0:loop_body/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask*
new_axis_maskk
loop_body/rotation_matrix/Sin_2Sinloop_body/stateful_uniform:z:0*
T0*
_output_shapes
:
/loop_body/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        
1loop_body/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
1loop_body/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ý
)loop_body/rotation_matrix/strided_slice_2StridedSlice#loop_body/rotation_matrix/Sin_2:y:08loop_body/rotation_matrix/strided_slice_2/stack:output:0:loop_body/rotation_matrix/strided_slice_2/stack_1:output:0:loop_body/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask*
new_axis_mask
loop_body/rotation_matrix/NegNeg2loop_body/rotation_matrix/strided_slice_2:output:0*
T0*
_output_shapes

:
/loop_body/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        
1loop_body/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
1loop_body/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÿ
)loop_body/rotation_matrix/strided_slice_3StridedSlice%loop_body/rotation_matrix/truediv:z:08loop_body/rotation_matrix/strided_slice_3/stack:output:0:loop_body/rotation_matrix/strided_slice_3/stack_1:output:0:loop_body/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask*
new_axis_maskk
loop_body/rotation_matrix/Sin_3Sinloop_body/stateful_uniform:z:0*
T0*
_output_shapes
:
/loop_body/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        
1loop_body/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
1loop_body/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ý
)loop_body/rotation_matrix/strided_slice_4StridedSlice#loop_body/rotation_matrix/Sin_3:y:08loop_body/rotation_matrix/strided_slice_4/stack:output:0:loop_body/rotation_matrix/strided_slice_4/stack_1:output:0:loop_body/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask*
new_axis_maskk
loop_body/rotation_matrix/Cos_3Cosloop_body/stateful_uniform:z:0*
T0*
_output_shapes
:
/loop_body/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        
1loop_body/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
1loop_body/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ý
)loop_body/rotation_matrix/strided_slice_5StridedSlice#loop_body/rotation_matrix/Cos_3:y:08loop_body/rotation_matrix/strided_slice_5/stack:output:0:loop_body/rotation_matrix/strided_slice_5/stack_1:output:0:loop_body/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask*
new_axis_mask
/loop_body/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        
1loop_body/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
1loop_body/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
)loop_body/rotation_matrix/strided_slice_6StridedSlice'loop_body/rotation_matrix/truediv_1:z:08loop_body/rotation_matrix/strided_slice_6/stack:output:0:loop_body/rotation_matrix/strided_slice_6/stack_1:output:0:loop_body/rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask*
new_axis_maskj
(loop_body/rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Á
&loop_body/rotation_matrix/zeros/packedPack0loop_body/rotation_matrix/strided_slice:output:01loop_body/rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:j
%loop_body/rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ±
loop_body/rotation_matrix/zerosFill/loop_body/rotation_matrix/zeros/packed:output:0.loop_body/rotation_matrix/zeros/Const:output:0*
T0*
_output_shapes

:g
%loop_body/rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ß
 loop_body/rotation_matrix/concatConcatV22loop_body/rotation_matrix/strided_slice_1:output:0!loop_body/rotation_matrix/Neg:y:02loop_body/rotation_matrix/strided_slice_3:output:02loop_body/rotation_matrix/strided_slice_4:output:02loop_body/rotation_matrix/strided_slice_5:output:02loop_body/rotation_matrix/strided_slice_6:output:0(loop_body/rotation_matrix/zeros:output:0.loop_body/rotation_matrix/concat/axis:output:0*
N*
T0*
_output_shapes

:r
loop_body/transform/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            q
'loop_body/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:s
)loop_body/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)loop_body/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¡
!loop_body/transform/strided_sliceStridedSlice"loop_body/transform/Shape:output:00loop_body/transform/strided_slice/stack:output:02loop_body/transform/strided_slice/stack_1:output:02loop_body/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:c
loop_body/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ×
.loop_body/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3loop_body/ExpandDims:output:0)loop_body/rotation_matrix/concat:output:0*loop_body/transform/strided_slice:output:0'loop_body/transform/fill_value:output:0*(
_output_shapes
:*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR§
loop_body/SqueezeSqueezeCloop_body/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*$
_output_shapes
:*
squeeze_dims
 \
pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:g
pfor/ReshapeReshapeMax:output:0pfor/Reshape/shape:output:0*
T0*
_output_shapes
:R
pfor/range/startConst*
_output_shapes
: *
dtype0*
value	B : R
pfor/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :|

pfor/rangeRangepfor/range/start:output:0Max:output:0pfor/range/delta:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Bloop_body/stateful_uniform/RngReadAndSkip/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Dloop_body/stateful_uniform/RngReadAndSkip/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Dloop_body/stateful_uniform/RngReadAndSkip/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
<loop_body/stateful_uniform/RngReadAndSkip/pfor/strided_sliceStridedSlicepfor/Reshape:output:0Kloop_body/stateful_uniform/RngReadAndSkip/pfor/strided_slice/stack:output:0Mloop_body/stateful_uniform/RngReadAndSkip/pfor/strided_slice/stack_1:output:0Mloop_body/stateful_uniform/RngReadAndSkip/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Jloop_body/stateful_uniform/RngReadAndSkip/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¿
<loop_body/stateful_uniform/RngReadAndSkip/pfor/TensorArrayV2TensorListReserveSloop_body/stateful_uniform/RngReadAndSkip/pfor/TensorArrayV2/element_shape:output:0Eloop_body/stateful_uniform/RngReadAndSkip/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0	*

shape_type0:éèÐv
4loop_body/stateful_uniform/RngReadAndSkip/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
Gloop_body/stateful_uniform/RngReadAndSkip/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Aloop_body/stateful_uniform/RngReadAndSkip/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Î
4loop_body/stateful_uniform/RngReadAndSkip/pfor/whileWhileJloop_body/stateful_uniform/RngReadAndSkip/pfor/while/loop_counter:output:0Ploop_body/stateful_uniform/RngReadAndSkip/pfor/while/maximum_iterations:output:0=loop_body/stateful_uniform/RngReadAndSkip/pfor/Const:output:0Eloop_body/stateful_uniform/RngReadAndSkip/pfor/TensorArrayV2:handle:0Eloop_body/stateful_uniform/RngReadAndSkip/pfor/strided_slice:output:02loop_body_stateful_uniform_rngreadandskip_resource*loop_body/stateful_uniform/Cast/x:output:0%loop_body/stateful_uniform/Cast_1:y:0*^loop_body/stateful_uniform/RngReadAndSkip*
T

2*
_lower_using_switch_merge(*
_num_original_outputs*$
_output_shapes
: : : : : : : : * 
_read_only_resource_inputs
 *
_stateful_parallelism( *J
bodyBR@
>loop_body_stateful_uniform_RngReadAndSkip_pfor_while_body_5665*J
condBR@
>loop_body_stateful_uniform_RngReadAndSkip_pfor_while_cond_5664*#
output_shapes
: : : : : : : : y
6loop_body/stateful_uniform/RngReadAndSkip/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	  
Oloop_body/stateful_uniform/RngReadAndSkip/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
Aloop_body/stateful_uniform/RngReadAndSkip/pfor/TensorListConcatV2TensorListConcatV2=loop_body/stateful_uniform/RngReadAndSkip/pfor/while:output:3Xloop_body/stateful_uniform/RngReadAndSkip/pfor/TensorListConcatV2/element_shape:output:0?loop_body/stateful_uniform/RngReadAndSkip/pfor/Const_1:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0	*

shape_type0
=loop_body/stateful_uniform/strided_slice/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: {
9loop_body/stateful_uniform/strided_slice/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ³
4loop_body/stateful_uniform/strided_slice/pfor/concatConcatV2Floop_body/stateful_uniform/strided_slice/pfor/concat/values_0:output:07loop_body/stateful_uniform/strided_slice/stack:output:0Bloop_body/stateful_uniform/strided_slice/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
?loop_body/stateful_uniform/strided_slice/pfor/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB: }
;loop_body/stateful_uniform/strided_slice/pfor/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : »
6loop_body/stateful_uniform/strided_slice/pfor/concat_1ConcatV2Hloop_body/stateful_uniform/strided_slice/pfor/concat_1/values_0:output:09loop_body/stateful_uniform/strided_slice/stack_1:output:0Dloop_body/stateful_uniform/strided_slice/pfor/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
?loop_body/stateful_uniform/strided_slice/pfor/concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB:}
;loop_body/stateful_uniform/strided_slice/pfor/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
6loop_body/stateful_uniform/strided_slice/pfor/concat_2ConcatV2Hloop_body/stateful_uniform/strided_slice/pfor/concat_2/values_0:output:09loop_body/stateful_uniform/strided_slice/stack_2:output:0Dloop_body/stateful_uniform/strided_slice/pfor/concat_2/axis:output:0*
N*
T0*
_output_shapes
:¸
:loop_body/stateful_uniform/strided_slice/pfor/StridedSliceStridedSliceJloop_body/stateful_uniform/RngReadAndSkip/pfor/TensorListConcatV2:tensor:0=loop_body/stateful_uniform/strided_slice/pfor/concat:output:0?loop_body/stateful_uniform/strided_slice/pfor/concat_1:output:0?loop_body/stateful_uniform/strided_slice/pfor/concat_2:output:0*
Index0*
T0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask
;loop_body/stateful_uniform/Bitcast/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=loop_body/stateful_uniform/Bitcast/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
=loop_body/stateful_uniform/Bitcast/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ø
5loop_body/stateful_uniform/Bitcast/pfor/strided_sliceStridedSlicepfor/Reshape:output:0Dloop_body/stateful_uniform/Bitcast/pfor/strided_slice/stack:output:0Floop_body/stateful_uniform/Bitcast/pfor/strided_slice/stack_1:output:0Floop_body/stateful_uniform/Bitcast/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Cloop_body/stateful_uniform/Bitcast/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿª
5loop_body/stateful_uniform/Bitcast/pfor/TensorArrayV2TensorListReserveLloop_body/stateful_uniform/Bitcast/pfor/TensorArrayV2/element_shape:output:0>loop_body/stateful_uniform/Bitcast/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÌo
-loop_body/stateful_uniform/Bitcast/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
@loop_body/stateful_uniform/Bitcast/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ|
:loop_body/stateful_uniform/Bitcast/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : É
-loop_body/stateful_uniform/Bitcast/pfor/whileStatelessWhileCloop_body/stateful_uniform/Bitcast/pfor/while/loop_counter:output:0Iloop_body/stateful_uniform/Bitcast/pfor/while/maximum_iterations:output:06loop_body/stateful_uniform/Bitcast/pfor/Const:output:0>loop_body/stateful_uniform/Bitcast/pfor/TensorArrayV2:handle:0>loop_body/stateful_uniform/Bitcast/pfor/strided_slice:output:0Cloop_body/stateful_uniform/strided_slice/pfor/StridedSlice:output:0*
T

2	*
_lower_using_switch_merge(*
_num_original_outputs*1
_output_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *
_stateful_parallelism( *C
body;R9
7loop_body_stateful_uniform_Bitcast_pfor_while_body_5730*C
cond;R9
7loop_body_stateful_uniform_Bitcast_pfor_while_cond_5729*0
output_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿr
/loop_body/stateful_uniform/Bitcast/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 
Hloop_body/stateful_uniform/Bitcast/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ô
:loop_body/stateful_uniform/Bitcast/pfor/TensorListConcatV2TensorListConcatV26loop_body/stateful_uniform/Bitcast/pfor/while:output:3Qloop_body/stateful_uniform/Bitcast/pfor/TensorListConcatV2/element_shape:output:08loop_body/stateful_uniform/Bitcast/pfor/Const_1:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0
?loop_body/stateful_uniform/strided_slice_1/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: }
;loop_body/stateful_uniform/strided_slice_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : »
6loop_body/stateful_uniform/strided_slice_1/pfor/concatConcatV2Hloop_body/stateful_uniform/strided_slice_1/pfor/concat/values_0:output:09loop_body/stateful_uniform/strided_slice_1/stack:output:0Dloop_body/stateful_uniform/strided_slice_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
Aloop_body/stateful_uniform/strided_slice_1/pfor/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB: 
=loop_body/stateful_uniform/strided_slice_1/pfor/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ã
8loop_body/stateful_uniform/strided_slice_1/pfor/concat_1ConcatV2Jloop_body/stateful_uniform/strided_slice_1/pfor/concat_1/values_0:output:0;loop_body/stateful_uniform/strided_slice_1/stack_1:output:0Floop_body/stateful_uniform/strided_slice_1/pfor/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
Aloop_body/stateful_uniform/strided_slice_1/pfor/concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB:
=loop_body/stateful_uniform/strided_slice_1/pfor/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ã
8loop_body/stateful_uniform/strided_slice_1/pfor/concat_2ConcatV2Jloop_body/stateful_uniform/strided_slice_1/pfor/concat_2/values_0:output:0;loop_body/stateful_uniform/strided_slice_1/stack_2:output:0Floop_body/stateful_uniform/strided_slice_1/pfor/concat_2/axis:output:0*
N*
T0*
_output_shapes
:À
<loop_body/stateful_uniform/strided_slice_1/pfor/StridedSliceStridedSliceJloop_body/stateful_uniform/RngReadAndSkip/pfor/TensorListConcatV2:tensor:0?loop_body/stateful_uniform/strided_slice_1/pfor/concat:output:0Aloop_body/stateful_uniform/strided_slice_1/pfor/concat_1:output:0Aloop_body/stateful_uniform/strided_slice_1/pfor/concat_2:output:0*
Index0*
T0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask
=loop_body/stateful_uniform/Bitcast_1/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?loop_body/stateful_uniform/Bitcast_1/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?loop_body/stateful_uniform/Bitcast_1/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
7loop_body/stateful_uniform/Bitcast_1/pfor/strided_sliceStridedSlicepfor/Reshape:output:0Floop_body/stateful_uniform/Bitcast_1/pfor/strided_slice/stack:output:0Hloop_body/stateful_uniform/Bitcast_1/pfor/strided_slice/stack_1:output:0Hloop_body/stateful_uniform/Bitcast_1/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Eloop_body/stateful_uniform/Bitcast_1/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ°
7loop_body/stateful_uniform/Bitcast_1/pfor/TensorArrayV2TensorListReserveNloop_body/stateful_uniform/Bitcast_1/pfor/TensorArrayV2/element_shape:output:0@loop_body/stateful_uniform/Bitcast_1/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÌq
/loop_body/stateful_uniform/Bitcast_1/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
Bloop_body/stateful_uniform/Bitcast_1/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ~
<loop_body/stateful_uniform/Bitcast_1/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Û
/loop_body/stateful_uniform/Bitcast_1/pfor/whileStatelessWhileEloop_body/stateful_uniform/Bitcast_1/pfor/while/loop_counter:output:0Kloop_body/stateful_uniform/Bitcast_1/pfor/while/maximum_iterations:output:08loop_body/stateful_uniform/Bitcast_1/pfor/Const:output:0@loop_body/stateful_uniform/Bitcast_1/pfor/TensorArrayV2:handle:0@loop_body/stateful_uniform/Bitcast_1/pfor/strided_slice:output:0Eloop_body/stateful_uniform/strided_slice_1/pfor/StridedSlice:output:0*
T

2	*
_lower_using_switch_merge(*
_num_original_outputs*1
_output_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *
_stateful_parallelism( *E
body=R;
9loop_body_stateful_uniform_Bitcast_1_pfor_while_body_5797*E
cond=R;
9loop_body_stateful_uniform_Bitcast_1_pfor_while_cond_5796*0
output_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿt
1loop_body/stateful_uniform/Bitcast_1/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 
Jloop_body/stateful_uniform/Bitcast_1/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ü
<loop_body/stateful_uniform/Bitcast_1/pfor/TensorListConcatV2TensorListConcatV28loop_body/stateful_uniform/Bitcast_1/pfor/while:output:3Sloop_body/stateful_uniform/Bitcast_1/pfor/TensorListConcatV2/element_shape:output:0:loop_body/stateful_uniform/Bitcast_1/pfor/Const_1:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0
Lloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Nloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Nloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¼
Floop_body/stateful_uniform/StatelessRandomUniformV2/pfor/strided_sliceStridedSlicepfor/Reshape:output:0Uloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/strided_slice/stack:output:0Wloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/strided_slice/stack_1:output:0Wloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Tloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÝ
Floop_body/stateful_uniform/StatelessRandomUniformV2/pfor/TensorArrayV2TensorListReserve]loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/TensorArrayV2/element_shape:output:0Oloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
>loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
Qloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Kloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¾	
>loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/whileStatelessWhileTloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/loop_counter:output:0Zloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/maximum_iterations:output:0Gloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/Const:output:0Oloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/TensorArrayV2:handle:0Oloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/strided_slice:output:0Eloop_body/stateful_uniform/Bitcast_1/pfor/TensorListConcatV2:tensor:0Cloop_body/stateful_uniform/Bitcast/pfor/TensorListConcatV2:tensor:0)loop_body/stateful_uniform/shape:output:0@loop_body/stateful_uniform/StatelessRandomUniformV2/alg:output:0*
T
2	*
_lower_using_switch_merge(*
_num_original_outputs	*L
_output_shapes:
8: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: * 
_read_only_resource_inputs
 *
_stateful_parallelism( *T
bodyLRJ
Hloop_body_stateful_uniform_StatelessRandomUniformV2_pfor_while_body_5854*T
condLRJ
Hloop_body_stateful_uniform_StatelessRandomUniformV2_pfor_while_cond_5853*K
output_shapes:
8: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: 
@loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 ª
Yloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¸
Kloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/TensorListConcatV2TensorListConcatV2Gloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while:output:3bloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/TensorListConcatV2/element_shape:output:0Iloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/Const_1:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0j
(loop_body/stateful_uniform/mul/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :l
*loop_body/stateful_uniform/mul/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : k
)loop_body/stateful_uniform/mul/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :º
'loop_body/stateful_uniform/mul/pfor/addAddV23loop_body/stateful_uniform/mul/pfor/Rank_1:output:02loop_body/stateful_uniform/mul/pfor/add/y:output:0*
T0*
_output_shapes
: ·
+loop_body/stateful_uniform/mul/pfor/MaximumMaximum+loop_body/stateful_uniform/mul/pfor/add:z:01loop_body/stateful_uniform/mul/pfor/Rank:output:0*
T0*
_output_shapes
: ­
)loop_body/stateful_uniform/mul/pfor/ShapeShapeTloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/TensorListConcatV2:tensor:0*
T0*
_output_shapes
:³
'loop_body/stateful_uniform/mul/pfor/subSub/loop_body/stateful_uniform/mul/pfor/Maximum:z:01loop_body/stateful_uniform/mul/pfor/Rank:output:0*
T0*
_output_shapes
: {
1loop_body/stateful_uniform/mul/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:Ä
+loop_body/stateful_uniform/mul/pfor/ReshapeReshape+loop_body/stateful_uniform/mul/pfor/sub:z:0:loop_body/stateful_uniform/mul/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:x
.loop_body/stateful_uniform/mul/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:Â
(loop_body/stateful_uniform/mul/pfor/TileTile7loop_body/stateful_uniform/mul/pfor/Tile/input:output:04loop_body/stateful_uniform/mul/pfor/Reshape:output:0*
T0*
_output_shapes
: 
7loop_body/stateful_uniform/mul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9loop_body/stateful_uniform/mul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9loop_body/stateful_uniform/mul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1loop_body/stateful_uniform/mul/pfor/strided_sliceStridedSlice2loop_body/stateful_uniform/mul/pfor/Shape:output:0@loop_body/stateful_uniform/mul/pfor/strided_slice/stack:output:0Bloop_body/stateful_uniform/mul/pfor/strided_slice/stack_1:output:0Bloop_body/stateful_uniform/mul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
9loop_body/stateful_uniform/mul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
;loop_body/stateful_uniform/mul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
;loop_body/stateful_uniform/mul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
3loop_body/stateful_uniform/mul/pfor/strided_slice_1StridedSlice2loop_body/stateful_uniform/mul/pfor/Shape:output:0Bloop_body/stateful_uniform/mul/pfor/strided_slice_1/stack:output:0Dloop_body/stateful_uniform/mul/pfor/strided_slice_1/stack_1:output:0Dloop_body/stateful_uniform/mul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskq
/loop_body/stateful_uniform/mul/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ë
*loop_body/stateful_uniform/mul/pfor/concatConcatV2:loop_body/stateful_uniform/mul/pfor/strided_slice:output:01loop_body/stateful_uniform/mul/pfor/Tile:output:0<loop_body/stateful_uniform/mul/pfor/strided_slice_1:output:08loop_body/stateful_uniform/mul/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:õ
-loop_body/stateful_uniform/mul/pfor/Reshape_1ReshapeTloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/TensorListConcatV2:tensor:03loop_body/stateful_uniform/mul/pfor/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
'loop_body/stateful_uniform/mul/pfor/MulMul6loop_body/stateful_uniform/mul/pfor/Reshape_1:output:0"loop_body/stateful_uniform/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
$loop_body/stateful_uniform/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :h
&loop_body/stateful_uniform/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : g
%loop_body/stateful_uniform/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :®
#loop_body/stateful_uniform/pfor/addAddV2/loop_body/stateful_uniform/pfor/Rank_1:output:0.loop_body/stateful_uniform/pfor/add/y:output:0*
T0*
_output_shapes
: «
'loop_body/stateful_uniform/pfor/MaximumMaximum'loop_body/stateful_uniform/pfor/add:z:0-loop_body/stateful_uniform/pfor/Rank:output:0*
T0*
_output_shapes
: 
%loop_body/stateful_uniform/pfor/ShapeShape+loop_body/stateful_uniform/mul/pfor/Mul:z:0*
T0*
_output_shapes
:§
#loop_body/stateful_uniform/pfor/subSub+loop_body/stateful_uniform/pfor/Maximum:z:0-loop_body/stateful_uniform/pfor/Rank:output:0*
T0*
_output_shapes
: w
-loop_body/stateful_uniform/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:¸
'loop_body/stateful_uniform/pfor/ReshapeReshape'loop_body/stateful_uniform/pfor/sub:z:06loop_body/stateful_uniform/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:t
*loop_body/stateful_uniform/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:¶
$loop_body/stateful_uniform/pfor/TileTile3loop_body/stateful_uniform/pfor/Tile/input:output:00loop_body/stateful_uniform/pfor/Reshape:output:0*
T0*
_output_shapes
: }
3loop_body/stateful_uniform/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5loop_body/stateful_uniform/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5loop_body/stateful_uniform/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
-loop_body/stateful_uniform/pfor/strided_sliceStridedSlice.loop_body/stateful_uniform/pfor/Shape:output:0<loop_body/stateful_uniform/pfor/strided_slice/stack:output:0>loop_body/stateful_uniform/pfor/strided_slice/stack_1:output:0>loop_body/stateful_uniform/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
5loop_body/stateful_uniform/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
7loop_body/stateful_uniform/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
7loop_body/stateful_uniform/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:õ
/loop_body/stateful_uniform/pfor/strided_slice_1StridedSlice.loop_body/stateful_uniform/pfor/Shape:output:0>loop_body/stateful_uniform/pfor/strided_slice_1/stack:output:0@loop_body/stateful_uniform/pfor/strided_slice_1/stack_1:output:0@loop_body/stateful_uniform/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskm
+loop_body/stateful_uniform/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ·
&loop_body/stateful_uniform/pfor/concatConcatV26loop_body/stateful_uniform/pfor/strided_slice:output:0-loop_body/stateful_uniform/pfor/Tile:output:08loop_body/stateful_uniform/pfor/strided_slice_1:output:04loop_body/stateful_uniform/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Ä
)loop_body/stateful_uniform/pfor/Reshape_1Reshape+loop_body/stateful_uniform/mul/pfor/Mul:z:0/loop_body/stateful_uniform/pfor/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
%loop_body/stateful_uniform/pfor/AddV2AddV22loop_body/stateful_uniform/pfor/Reshape_1:output:0'loop_body/stateful_uniform/min:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(loop_body/rotation_matrix/Cos_1/pfor/CosCos)loop_body/stateful_uniform/pfor/AddV2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
)loop_body/rotation_matrix/mul_3/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :m
+loop_body/rotation_matrix/mul_3/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : l
*loop_body/rotation_matrix/mul_3/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :½
(loop_body/rotation_matrix/mul_3/pfor/addAddV24loop_body/rotation_matrix/mul_3/pfor/Rank_1:output:03loop_body/rotation_matrix/mul_3/pfor/add/y:output:0*
T0*
_output_shapes
: º
,loop_body/rotation_matrix/mul_3/pfor/MaximumMaximum,loop_body/rotation_matrix/mul_3/pfor/add:z:02loop_body/rotation_matrix/mul_3/pfor/Rank:output:0*
T0*
_output_shapes
: 
*loop_body/rotation_matrix/mul_3/pfor/ShapeShape,loop_body/rotation_matrix/Cos_1/pfor/Cos:y:0*
T0*
_output_shapes
:¶
(loop_body/rotation_matrix/mul_3/pfor/subSub0loop_body/rotation_matrix/mul_3/pfor/Maximum:z:02loop_body/rotation_matrix/mul_3/pfor/Rank:output:0*
T0*
_output_shapes
: |
2loop_body/rotation_matrix/mul_3/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:Ç
,loop_body/rotation_matrix/mul_3/pfor/ReshapeReshape,loop_body/rotation_matrix/mul_3/pfor/sub:z:0;loop_body/rotation_matrix/mul_3/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:y
/loop_body/rotation_matrix/mul_3/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:Å
)loop_body/rotation_matrix/mul_3/pfor/TileTile8loop_body/rotation_matrix/mul_3/pfor/Tile/input:output:05loop_body/rotation_matrix/mul_3/pfor/Reshape:output:0*
T0*
_output_shapes
: 
8loop_body/rotation_matrix/mul_3/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
:loop_body/rotation_matrix/mul_3/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
:loop_body/rotation_matrix/mul_3/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
2loop_body/rotation_matrix/mul_3/pfor/strided_sliceStridedSlice3loop_body/rotation_matrix/mul_3/pfor/Shape:output:0Aloop_body/rotation_matrix/mul_3/pfor/strided_slice/stack:output:0Cloop_body/rotation_matrix/mul_3/pfor/strided_slice/stack_1:output:0Cloop_body/rotation_matrix/mul_3/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
:loop_body/rotation_matrix/mul_3/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
<loop_body/rotation_matrix/mul_3/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
<loop_body/rotation_matrix/mul_3/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
4loop_body/rotation_matrix/mul_3/pfor/strided_slice_1StridedSlice3loop_body/rotation_matrix/mul_3/pfor/Shape:output:0Cloop_body/rotation_matrix/mul_3/pfor/strided_slice_1/stack:output:0Eloop_body/rotation_matrix/mul_3/pfor/strided_slice_1/stack_1:output:0Eloop_body/rotation_matrix/mul_3/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskr
0loop_body/rotation_matrix/mul_3/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ð
+loop_body/rotation_matrix/mul_3/pfor/concatConcatV2;loop_body/rotation_matrix/mul_3/pfor/strided_slice:output:02loop_body/rotation_matrix/mul_3/pfor/Tile:output:0=loop_body/rotation_matrix/mul_3/pfor/strided_slice_1:output:09loop_body/rotation_matrix/mul_3/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Ï
.loop_body/rotation_matrix/mul_3/pfor/Reshape_1Reshape,loop_body/rotation_matrix/Cos_1/pfor/Cos:y:04loop_body/rotation_matrix/mul_3/pfor/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
(loop_body/rotation_matrix/mul_3/pfor/MulMul7loop_body/rotation_matrix/mul_3/pfor/Reshape_1:output:0#loop_body/rotation_matrix/sub_7:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(loop_body/rotation_matrix/Sin_1/pfor/SinSin)loop_body/stateful_uniform/pfor/AddV2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
)loop_body/rotation_matrix/mul_2/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :m
+loop_body/rotation_matrix/mul_2/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : l
*loop_body/rotation_matrix/mul_2/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :½
(loop_body/rotation_matrix/mul_2/pfor/addAddV24loop_body/rotation_matrix/mul_2/pfor/Rank_1:output:03loop_body/rotation_matrix/mul_2/pfor/add/y:output:0*
T0*
_output_shapes
: º
,loop_body/rotation_matrix/mul_2/pfor/MaximumMaximum,loop_body/rotation_matrix/mul_2/pfor/add:z:02loop_body/rotation_matrix/mul_2/pfor/Rank:output:0*
T0*
_output_shapes
: 
*loop_body/rotation_matrix/mul_2/pfor/ShapeShape,loop_body/rotation_matrix/Sin_1/pfor/Sin:y:0*
T0*
_output_shapes
:¶
(loop_body/rotation_matrix/mul_2/pfor/subSub0loop_body/rotation_matrix/mul_2/pfor/Maximum:z:02loop_body/rotation_matrix/mul_2/pfor/Rank:output:0*
T0*
_output_shapes
: |
2loop_body/rotation_matrix/mul_2/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:Ç
,loop_body/rotation_matrix/mul_2/pfor/ReshapeReshape,loop_body/rotation_matrix/mul_2/pfor/sub:z:0;loop_body/rotation_matrix/mul_2/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:y
/loop_body/rotation_matrix/mul_2/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:Å
)loop_body/rotation_matrix/mul_2/pfor/TileTile8loop_body/rotation_matrix/mul_2/pfor/Tile/input:output:05loop_body/rotation_matrix/mul_2/pfor/Reshape:output:0*
T0*
_output_shapes
: 
8loop_body/rotation_matrix/mul_2/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
:loop_body/rotation_matrix/mul_2/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
:loop_body/rotation_matrix/mul_2/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
2loop_body/rotation_matrix/mul_2/pfor/strided_sliceStridedSlice3loop_body/rotation_matrix/mul_2/pfor/Shape:output:0Aloop_body/rotation_matrix/mul_2/pfor/strided_slice/stack:output:0Cloop_body/rotation_matrix/mul_2/pfor/strided_slice/stack_1:output:0Cloop_body/rotation_matrix/mul_2/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
:loop_body/rotation_matrix/mul_2/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
<loop_body/rotation_matrix/mul_2/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
<loop_body/rotation_matrix/mul_2/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
4loop_body/rotation_matrix/mul_2/pfor/strided_slice_1StridedSlice3loop_body/rotation_matrix/mul_2/pfor/Shape:output:0Cloop_body/rotation_matrix/mul_2/pfor/strided_slice_1/stack:output:0Eloop_body/rotation_matrix/mul_2/pfor/strided_slice_1/stack_1:output:0Eloop_body/rotation_matrix/mul_2/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskr
0loop_body/rotation_matrix/mul_2/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ð
+loop_body/rotation_matrix/mul_2/pfor/concatConcatV2;loop_body/rotation_matrix/mul_2/pfor/strided_slice:output:02loop_body/rotation_matrix/mul_2/pfor/Tile:output:0=loop_body/rotation_matrix/mul_2/pfor/strided_slice_1:output:09loop_body/rotation_matrix/mul_2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Ï
.loop_body/rotation_matrix/mul_2/pfor/Reshape_1Reshape,loop_body/rotation_matrix/Sin_1/pfor/Sin:y:04loop_body/rotation_matrix/mul_2/pfor/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
(loop_body/rotation_matrix/mul_2/pfor/MulMul7loop_body/rotation_matrix/mul_2/pfor/Reshape_1:output:0#loop_body/rotation_matrix/sub_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
'loop_body/rotation_matrix/add/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :k
)loop_body/rotation_matrix/add/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :¼
*loop_body/rotation_matrix/add/pfor/MaximumMaximum2loop_body/rotation_matrix/add/pfor/Rank_1:output:00loop_body/rotation_matrix/add/pfor/Rank:output:0*
T0*
_output_shapes
: 
(loop_body/rotation_matrix/add/pfor/ShapeShape,loop_body/rotation_matrix/mul_2/pfor/Mul:z:0*
T0*
_output_shapes
:°
&loop_body/rotation_matrix/add/pfor/subSub.loop_body/rotation_matrix/add/pfor/Maximum:z:00loop_body/rotation_matrix/add/pfor/Rank:output:0*
T0*
_output_shapes
: z
0loop_body/rotation_matrix/add/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:Á
*loop_body/rotation_matrix/add/pfor/ReshapeReshape*loop_body/rotation_matrix/add/pfor/sub:z:09loop_body/rotation_matrix/add/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:w
-loop_body/rotation_matrix/add/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:¿
'loop_body/rotation_matrix/add/pfor/TileTile6loop_body/rotation_matrix/add/pfor/Tile/input:output:03loop_body/rotation_matrix/add/pfor/Reshape:output:0*
T0*
_output_shapes
: 
6loop_body/rotation_matrix/add/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
8loop_body/rotation_matrix/add/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
8loop_body/rotation_matrix/add/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:þ
0loop_body/rotation_matrix/add/pfor/strided_sliceStridedSlice1loop_body/rotation_matrix/add/pfor/Shape:output:0?loop_body/rotation_matrix/add/pfor/strided_slice/stack:output:0Aloop_body/rotation_matrix/add/pfor/strided_slice/stack_1:output:0Aloop_body/rotation_matrix/add/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
8loop_body/rotation_matrix/add/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
:loop_body/rotation_matrix/add/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
:loop_body/rotation_matrix/add/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
2loop_body/rotation_matrix/add/pfor/strided_slice_1StridedSlice1loop_body/rotation_matrix/add/pfor/Shape:output:0Aloop_body/rotation_matrix/add/pfor/strided_slice_1/stack:output:0Cloop_body/rotation_matrix/add/pfor/strided_slice_1/stack_1:output:0Cloop_body/rotation_matrix/add/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskp
.loop_body/rotation_matrix/add/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Æ
)loop_body/rotation_matrix/add/pfor/concatConcatV29loop_body/rotation_matrix/add/pfor/strided_slice:output:00loop_body/rotation_matrix/add/pfor/Tile:output:0;loop_body/rotation_matrix/add/pfor/strided_slice_1:output:07loop_body/rotation_matrix/add/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Ë
,loop_body/rotation_matrix/add/pfor/Reshape_1Reshape,loop_body/rotation_matrix/mul_2/pfor/Mul:z:02loop_body/rotation_matrix/add/pfor/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*loop_body/rotation_matrix/add/pfor/Shape_1Shape,loop_body/rotation_matrix/mul_3/pfor/Mul:z:0*
T0*
_output_shapes
:´
(loop_body/rotation_matrix/add/pfor/sub_1Sub.loop_body/rotation_matrix/add/pfor/Maximum:z:02loop_body/rotation_matrix/add/pfor/Rank_1:output:0*
T0*
_output_shapes
: |
2loop_body/rotation_matrix/add/pfor/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB:Ç
,loop_body/rotation_matrix/add/pfor/Reshape_2Reshape,loop_body/rotation_matrix/add/pfor/sub_1:z:0;loop_body/rotation_matrix/add/pfor/Reshape_2/shape:output:0*
T0*
_output_shapes
:y
/loop_body/rotation_matrix/add/pfor/Tile_1/inputConst*
_output_shapes
:*
dtype0*
valueB:Å
)loop_body/rotation_matrix/add/pfor/Tile_1Tile8loop_body/rotation_matrix/add/pfor/Tile_1/input:output:05loop_body/rotation_matrix/add/pfor/Reshape_2:output:0*
T0*
_output_shapes
: 
8loop_body/rotation_matrix/add/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
:loop_body/rotation_matrix/add/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
:loop_body/rotation_matrix/add/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
2loop_body/rotation_matrix/add/pfor/strided_slice_2StridedSlice3loop_body/rotation_matrix/add/pfor/Shape_1:output:0Aloop_body/rotation_matrix/add/pfor/strided_slice_2/stack:output:0Cloop_body/rotation_matrix/add/pfor/strided_slice_2/stack_1:output:0Cloop_body/rotation_matrix/add/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
8loop_body/rotation_matrix/add/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
:loop_body/rotation_matrix/add/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
:loop_body/rotation_matrix/add/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
2loop_body/rotation_matrix/add/pfor/strided_slice_3StridedSlice3loop_body/rotation_matrix/add/pfor/Shape_1:output:0Aloop_body/rotation_matrix/add/pfor/strided_slice_3/stack:output:0Cloop_body/rotation_matrix/add/pfor/strided_slice_3/stack_1:output:0Cloop_body/rotation_matrix/add/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskr
0loop_body/rotation_matrix/add/pfor/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Î
+loop_body/rotation_matrix/add/pfor/concat_1ConcatV2;loop_body/rotation_matrix/add/pfor/strided_slice_2:output:02loop_body/rotation_matrix/add/pfor/Tile_1:output:0;loop_body/rotation_matrix/add/pfor/strided_slice_3:output:09loop_body/rotation_matrix/add/pfor/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Í
,loop_body/rotation_matrix/add/pfor/Reshape_3Reshape,loop_body/rotation_matrix/mul_3/pfor/Mul:z:04loop_body/rotation_matrix/add/pfor/concat_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
(loop_body/rotation_matrix/add/pfor/AddV2AddV25loop_body/rotation_matrix/add/pfor/Reshape_1:output:05loop_body/rotation_matrix/add/pfor/Reshape_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
)loop_body/rotation_matrix/sub_8/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B : l
*loop_body/rotation_matrix/sub_8/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :»
(loop_body/rotation_matrix/sub_8/pfor/addAddV22loop_body/rotation_matrix/sub_8/pfor/Rank:output:03loop_body/rotation_matrix/sub_8/pfor/add/y:output:0*
T0*
_output_shapes
: m
+loop_body/rotation_matrix/sub_8/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :¼
,loop_body/rotation_matrix/sub_8/pfor/MaximumMaximum4loop_body/rotation_matrix/sub_8/pfor/Rank_1:output:0,loop_body/rotation_matrix/sub_8/pfor/add:z:0*
T0*
_output_shapes
: 
*loop_body/rotation_matrix/sub_8/pfor/ShapeShape,loop_body/rotation_matrix/add/pfor/AddV2:z:0*
T0*
_output_shapes
:¸
(loop_body/rotation_matrix/sub_8/pfor/subSub0loop_body/rotation_matrix/sub_8/pfor/Maximum:z:04loop_body/rotation_matrix/sub_8/pfor/Rank_1:output:0*
T0*
_output_shapes
: |
2loop_body/rotation_matrix/sub_8/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:Ç
,loop_body/rotation_matrix/sub_8/pfor/ReshapeReshape,loop_body/rotation_matrix/sub_8/pfor/sub:z:0;loop_body/rotation_matrix/sub_8/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:y
/loop_body/rotation_matrix/sub_8/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:Å
)loop_body/rotation_matrix/sub_8/pfor/TileTile8loop_body/rotation_matrix/sub_8/pfor/Tile/input:output:05loop_body/rotation_matrix/sub_8/pfor/Reshape:output:0*
T0*
_output_shapes
: 
8loop_body/rotation_matrix/sub_8/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
:loop_body/rotation_matrix/sub_8/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
:loop_body/rotation_matrix/sub_8/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
2loop_body/rotation_matrix/sub_8/pfor/strided_sliceStridedSlice3loop_body/rotation_matrix/sub_8/pfor/Shape:output:0Aloop_body/rotation_matrix/sub_8/pfor/strided_slice/stack:output:0Cloop_body/rotation_matrix/sub_8/pfor/strided_slice/stack_1:output:0Cloop_body/rotation_matrix/sub_8/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
:loop_body/rotation_matrix/sub_8/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
<loop_body/rotation_matrix/sub_8/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
<loop_body/rotation_matrix/sub_8/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
4loop_body/rotation_matrix/sub_8/pfor/strided_slice_1StridedSlice3loop_body/rotation_matrix/sub_8/pfor/Shape:output:0Cloop_body/rotation_matrix/sub_8/pfor/strided_slice_1/stack:output:0Eloop_body/rotation_matrix/sub_8/pfor/strided_slice_1/stack_1:output:0Eloop_body/rotation_matrix/sub_8/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskr
0loop_body/rotation_matrix/sub_8/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ð
+loop_body/rotation_matrix/sub_8/pfor/concatConcatV2;loop_body/rotation_matrix/sub_8/pfor/strided_slice:output:02loop_body/rotation_matrix/sub_8/pfor/Tile:output:0=loop_body/rotation_matrix/sub_8/pfor/strided_slice_1:output:09loop_body/rotation_matrix/sub_8/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Ï
.loop_body/rotation_matrix/sub_8/pfor/Reshape_1Reshape,loop_body/rotation_matrix/add/pfor/AddV2:z:04loop_body/rotation_matrix/sub_8/pfor/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÁ
*loop_body/rotation_matrix/sub_8/pfor/Sub_1Sub#loop_body/rotation_matrix/sub_5:z:07loop_body/rotation_matrix/sub_8/pfor/Reshape_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
-loop_body/rotation_matrix/truediv_1/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :q
/loop_body/rotation_matrix/truediv_1/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : p
.loop_body/rotation_matrix/truediv_1/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :É
,loop_body/rotation_matrix/truediv_1/pfor/addAddV28loop_body/rotation_matrix/truediv_1/pfor/Rank_1:output:07loop_body/rotation_matrix/truediv_1/pfor/add/y:output:0*
T0*
_output_shapes
: Æ
0loop_body/rotation_matrix/truediv_1/pfor/MaximumMaximum0loop_body/rotation_matrix/truediv_1/pfor/add:z:06loop_body/rotation_matrix/truediv_1/pfor/Rank:output:0*
T0*
_output_shapes
: 
.loop_body/rotation_matrix/truediv_1/pfor/ShapeShape.loop_body/rotation_matrix/sub_8/pfor/Sub_1:z:0*
T0*
_output_shapes
:Â
,loop_body/rotation_matrix/truediv_1/pfor/subSub4loop_body/rotation_matrix/truediv_1/pfor/Maximum:z:06loop_body/rotation_matrix/truediv_1/pfor/Rank:output:0*
T0*
_output_shapes
: 
6loop_body/rotation_matrix/truediv_1/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:Ó
0loop_body/rotation_matrix/truediv_1/pfor/ReshapeReshape0loop_body/rotation_matrix/truediv_1/pfor/sub:z:0?loop_body/rotation_matrix/truediv_1/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:}
3loop_body/rotation_matrix/truediv_1/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:Ñ
-loop_body/rotation_matrix/truediv_1/pfor/TileTile<loop_body/rotation_matrix/truediv_1/pfor/Tile/input:output:09loop_body/rotation_matrix/truediv_1/pfor/Reshape:output:0*
T0*
_output_shapes
: 
<loop_body/rotation_matrix/truediv_1/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
>loop_body/rotation_matrix/truediv_1/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
>loop_body/rotation_matrix/truediv_1/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
6loop_body/rotation_matrix/truediv_1/pfor/strided_sliceStridedSlice7loop_body/rotation_matrix/truediv_1/pfor/Shape:output:0Eloop_body/rotation_matrix/truediv_1/pfor/strided_slice/stack:output:0Gloop_body/rotation_matrix/truediv_1/pfor/strided_slice/stack_1:output:0Gloop_body/rotation_matrix/truediv_1/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
>loop_body/rotation_matrix/truediv_1/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
@loop_body/rotation_matrix/truediv_1/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
@loop_body/rotation_matrix/truediv_1/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¢
8loop_body/rotation_matrix/truediv_1/pfor/strided_slice_1StridedSlice7loop_body/rotation_matrix/truediv_1/pfor/Shape:output:0Gloop_body/rotation_matrix/truediv_1/pfor/strided_slice_1/stack:output:0Iloop_body/rotation_matrix/truediv_1/pfor/strided_slice_1/stack_1:output:0Iloop_body/rotation_matrix/truediv_1/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskv
4loop_body/rotation_matrix/truediv_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ä
/loop_body/rotation_matrix/truediv_1/pfor/concatConcatV2?loop_body/rotation_matrix/truediv_1/pfor/strided_slice:output:06loop_body/rotation_matrix/truediv_1/pfor/Tile:output:0Aloop_body/rotation_matrix/truediv_1/pfor/strided_slice_1:output:0=loop_body/rotation_matrix/truediv_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Ù
2loop_body/rotation_matrix/truediv_1/pfor/Reshape_1Reshape.loop_body/rotation_matrix/sub_8/pfor/Sub_1:z:08loop_body/rotation_matrix/truediv_1/pfor/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
0loop_body/rotation_matrix/truediv_1/pfor/RealDivRealDiv;loop_body/rotation_matrix/truediv_1/pfor/Reshape_1:output:0.loop_body/rotation_matrix/truediv_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
>loop_body/rotation_matrix/strided_slice_6/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: |
:loop_body/rotation_matrix/strided_slice_6/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ·
5loop_body/rotation_matrix/strided_slice_6/pfor/concatConcatV2Gloop_body/rotation_matrix/strided_slice_6/pfor/concat/values_0:output:08loop_body/rotation_matrix/strided_slice_6/stack:output:0Cloop_body/rotation_matrix/strided_slice_6/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
@loop_body/rotation_matrix/strided_slice_6/pfor/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB: ~
<loop_body/rotation_matrix/strided_slice_6/pfor/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
7loop_body/rotation_matrix/strided_slice_6/pfor/concat_1ConcatV2Iloop_body/rotation_matrix/strided_slice_6/pfor/concat_1/values_0:output:0:loop_body/rotation_matrix/strided_slice_6/stack_1:output:0Eloop_body/rotation_matrix/strided_slice_6/pfor/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
@loop_body/rotation_matrix/strided_slice_6/pfor/concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB:~
<loop_body/rotation_matrix/strided_slice_6/pfor/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
7loop_body/rotation_matrix/strided_slice_6/pfor/concat_2ConcatV2Iloop_body/rotation_matrix/strided_slice_6/pfor/concat_2/values_0:output:0:loop_body/rotation_matrix/strided_slice_6/stack_2:output:0Eloop_body/rotation_matrix/strided_slice_6/pfor/concat_2/axis:output:0*
N*
T0*
_output_shapes
:¿
;loop_body/rotation_matrix/strided_slice_6/pfor/StridedSliceStridedSlice4loop_body/rotation_matrix/truediv_1/pfor/RealDiv:z:0>loop_body/rotation_matrix/strided_slice_6/pfor/concat:output:0@loop_body/rotation_matrix/strided_slice_6/pfor/concat_1:output:0@loop_body/rotation_matrix/strided_slice_6/pfor/concat_2:output:0*
Index0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask
(loop_body/rotation_matrix/Cos_3/pfor/CosCos)loop_body/stateful_uniform/pfor/AddV2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
>loop_body/rotation_matrix/strided_slice_5/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: |
:loop_body/rotation_matrix/strided_slice_5/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ·
5loop_body/rotation_matrix/strided_slice_5/pfor/concatConcatV2Gloop_body/rotation_matrix/strided_slice_5/pfor/concat/values_0:output:08loop_body/rotation_matrix/strided_slice_5/stack:output:0Cloop_body/rotation_matrix/strided_slice_5/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
@loop_body/rotation_matrix/strided_slice_5/pfor/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB: ~
<loop_body/rotation_matrix/strided_slice_5/pfor/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
7loop_body/rotation_matrix/strided_slice_5/pfor/concat_1ConcatV2Iloop_body/rotation_matrix/strided_slice_5/pfor/concat_1/values_0:output:0:loop_body/rotation_matrix/strided_slice_5/stack_1:output:0Eloop_body/rotation_matrix/strided_slice_5/pfor/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
@loop_body/rotation_matrix/strided_slice_5/pfor/concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB:~
<loop_body/rotation_matrix/strided_slice_5/pfor/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
7loop_body/rotation_matrix/strided_slice_5/pfor/concat_2ConcatV2Iloop_body/rotation_matrix/strided_slice_5/pfor/concat_2/values_0:output:0:loop_body/rotation_matrix/strided_slice_5/stack_2:output:0Eloop_body/rotation_matrix/strided_slice_5/pfor/concat_2/axis:output:0*
N*
T0*
_output_shapes
:·
;loop_body/rotation_matrix/strided_slice_5/pfor/StridedSliceStridedSlice,loop_body/rotation_matrix/Cos_3/pfor/Cos:y:0>loop_body/rotation_matrix/strided_slice_5/pfor/concat:output:0@loop_body/rotation_matrix/strided_slice_5/pfor/concat_1:output:0@loop_body/rotation_matrix/strided_slice_5/pfor/concat_2:output:0*
Index0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask
(loop_body/rotation_matrix/Sin_3/pfor/SinSin)loop_body/stateful_uniform/pfor/AddV2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
>loop_body/rotation_matrix/strided_slice_4/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: |
:loop_body/rotation_matrix/strided_slice_4/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ·
5loop_body/rotation_matrix/strided_slice_4/pfor/concatConcatV2Gloop_body/rotation_matrix/strided_slice_4/pfor/concat/values_0:output:08loop_body/rotation_matrix/strided_slice_4/stack:output:0Cloop_body/rotation_matrix/strided_slice_4/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
@loop_body/rotation_matrix/strided_slice_4/pfor/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB: ~
<loop_body/rotation_matrix/strided_slice_4/pfor/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
7loop_body/rotation_matrix/strided_slice_4/pfor/concat_1ConcatV2Iloop_body/rotation_matrix/strided_slice_4/pfor/concat_1/values_0:output:0:loop_body/rotation_matrix/strided_slice_4/stack_1:output:0Eloop_body/rotation_matrix/strided_slice_4/pfor/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
@loop_body/rotation_matrix/strided_slice_4/pfor/concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB:~
<loop_body/rotation_matrix/strided_slice_4/pfor/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
7loop_body/rotation_matrix/strided_slice_4/pfor/concat_2ConcatV2Iloop_body/rotation_matrix/strided_slice_4/pfor/concat_2/values_0:output:0:loop_body/rotation_matrix/strided_slice_4/stack_2:output:0Eloop_body/rotation_matrix/strided_slice_4/pfor/concat_2/axis:output:0*
N*
T0*
_output_shapes
:·
;loop_body/rotation_matrix/strided_slice_4/pfor/StridedSliceStridedSlice,loop_body/rotation_matrix/Sin_3/pfor/Sin:y:0>loop_body/rotation_matrix/strided_slice_4/pfor/concat:output:0@loop_body/rotation_matrix/strided_slice_4/pfor/concat_1:output:0@loop_body/rotation_matrix/strided_slice_4/pfor/concat_2:output:0*
Index0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask
&loop_body/rotation_matrix/Sin/pfor/SinSin)loop_body/stateful_uniform/pfor/AddV2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
)loop_body/rotation_matrix/mul_1/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :m
+loop_body/rotation_matrix/mul_1/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : l
*loop_body/rotation_matrix/mul_1/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :½
(loop_body/rotation_matrix/mul_1/pfor/addAddV24loop_body/rotation_matrix/mul_1/pfor/Rank_1:output:03loop_body/rotation_matrix/mul_1/pfor/add/y:output:0*
T0*
_output_shapes
: º
,loop_body/rotation_matrix/mul_1/pfor/MaximumMaximum,loop_body/rotation_matrix/mul_1/pfor/add:z:02loop_body/rotation_matrix/mul_1/pfor/Rank:output:0*
T0*
_output_shapes
: 
*loop_body/rotation_matrix/mul_1/pfor/ShapeShape*loop_body/rotation_matrix/Sin/pfor/Sin:y:0*
T0*
_output_shapes
:¶
(loop_body/rotation_matrix/mul_1/pfor/subSub0loop_body/rotation_matrix/mul_1/pfor/Maximum:z:02loop_body/rotation_matrix/mul_1/pfor/Rank:output:0*
T0*
_output_shapes
: |
2loop_body/rotation_matrix/mul_1/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:Ç
,loop_body/rotation_matrix/mul_1/pfor/ReshapeReshape,loop_body/rotation_matrix/mul_1/pfor/sub:z:0;loop_body/rotation_matrix/mul_1/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:y
/loop_body/rotation_matrix/mul_1/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:Å
)loop_body/rotation_matrix/mul_1/pfor/TileTile8loop_body/rotation_matrix/mul_1/pfor/Tile/input:output:05loop_body/rotation_matrix/mul_1/pfor/Reshape:output:0*
T0*
_output_shapes
: 
8loop_body/rotation_matrix/mul_1/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
:loop_body/rotation_matrix/mul_1/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
:loop_body/rotation_matrix/mul_1/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
2loop_body/rotation_matrix/mul_1/pfor/strided_sliceStridedSlice3loop_body/rotation_matrix/mul_1/pfor/Shape:output:0Aloop_body/rotation_matrix/mul_1/pfor/strided_slice/stack:output:0Cloop_body/rotation_matrix/mul_1/pfor/strided_slice/stack_1:output:0Cloop_body/rotation_matrix/mul_1/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
:loop_body/rotation_matrix/mul_1/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
<loop_body/rotation_matrix/mul_1/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
<loop_body/rotation_matrix/mul_1/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
4loop_body/rotation_matrix/mul_1/pfor/strided_slice_1StridedSlice3loop_body/rotation_matrix/mul_1/pfor/Shape:output:0Cloop_body/rotation_matrix/mul_1/pfor/strided_slice_1/stack:output:0Eloop_body/rotation_matrix/mul_1/pfor/strided_slice_1/stack_1:output:0Eloop_body/rotation_matrix/mul_1/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskr
0loop_body/rotation_matrix/mul_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ð
+loop_body/rotation_matrix/mul_1/pfor/concatConcatV2;loop_body/rotation_matrix/mul_1/pfor/strided_slice:output:02loop_body/rotation_matrix/mul_1/pfor/Tile:output:0=loop_body/rotation_matrix/mul_1/pfor/strided_slice_1:output:09loop_body/rotation_matrix/mul_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Í
.loop_body/rotation_matrix/mul_1/pfor/Reshape_1Reshape*loop_body/rotation_matrix/Sin/pfor/Sin:y:04loop_body/rotation_matrix/mul_1/pfor/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
(loop_body/rotation_matrix/mul_1/pfor/MulMul7loop_body/rotation_matrix/mul_1/pfor/Reshape_1:output:0#loop_body/rotation_matrix/sub_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&loop_body/rotation_matrix/Cos/pfor/CosCos)loop_body/stateful_uniform/pfor/AddV2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
'loop_body/rotation_matrix/mul/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :k
)loop_body/rotation_matrix/mul/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : j
(loop_body/rotation_matrix/mul/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :·
&loop_body/rotation_matrix/mul/pfor/addAddV22loop_body/rotation_matrix/mul/pfor/Rank_1:output:01loop_body/rotation_matrix/mul/pfor/add/y:output:0*
T0*
_output_shapes
: ´
*loop_body/rotation_matrix/mul/pfor/MaximumMaximum*loop_body/rotation_matrix/mul/pfor/add:z:00loop_body/rotation_matrix/mul/pfor/Rank:output:0*
T0*
_output_shapes
: 
(loop_body/rotation_matrix/mul/pfor/ShapeShape*loop_body/rotation_matrix/Cos/pfor/Cos:y:0*
T0*
_output_shapes
:°
&loop_body/rotation_matrix/mul/pfor/subSub.loop_body/rotation_matrix/mul/pfor/Maximum:z:00loop_body/rotation_matrix/mul/pfor/Rank:output:0*
T0*
_output_shapes
: z
0loop_body/rotation_matrix/mul/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:Á
*loop_body/rotation_matrix/mul/pfor/ReshapeReshape*loop_body/rotation_matrix/mul/pfor/sub:z:09loop_body/rotation_matrix/mul/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:w
-loop_body/rotation_matrix/mul/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:¿
'loop_body/rotation_matrix/mul/pfor/TileTile6loop_body/rotation_matrix/mul/pfor/Tile/input:output:03loop_body/rotation_matrix/mul/pfor/Reshape:output:0*
T0*
_output_shapes
: 
6loop_body/rotation_matrix/mul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
8loop_body/rotation_matrix/mul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
8loop_body/rotation_matrix/mul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:þ
0loop_body/rotation_matrix/mul/pfor/strided_sliceStridedSlice1loop_body/rotation_matrix/mul/pfor/Shape:output:0?loop_body/rotation_matrix/mul/pfor/strided_slice/stack:output:0Aloop_body/rotation_matrix/mul/pfor/strided_slice/stack_1:output:0Aloop_body/rotation_matrix/mul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
8loop_body/rotation_matrix/mul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
:loop_body/rotation_matrix/mul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
:loop_body/rotation_matrix/mul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
2loop_body/rotation_matrix/mul/pfor/strided_slice_1StridedSlice1loop_body/rotation_matrix/mul/pfor/Shape:output:0Aloop_body/rotation_matrix/mul/pfor/strided_slice_1/stack:output:0Cloop_body/rotation_matrix/mul/pfor/strided_slice_1/stack_1:output:0Cloop_body/rotation_matrix/mul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskp
.loop_body/rotation_matrix/mul/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Æ
)loop_body/rotation_matrix/mul/pfor/concatConcatV29loop_body/rotation_matrix/mul/pfor/strided_slice:output:00loop_body/rotation_matrix/mul/pfor/Tile:output:0;loop_body/rotation_matrix/mul/pfor/strided_slice_1:output:07loop_body/rotation_matrix/mul/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:É
,loop_body/rotation_matrix/mul/pfor/Reshape_1Reshape*loop_body/rotation_matrix/Cos/pfor/Cos:y:02loop_body/rotation_matrix/mul/pfor/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ»
&loop_body/rotation_matrix/mul/pfor/MulMul5loop_body/rotation_matrix/mul/pfor/Reshape_1:output:0#loop_body/rotation_matrix/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
)loop_body/rotation_matrix/sub_3/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :m
+loop_body/rotation_matrix/sub_3/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :Â
,loop_body/rotation_matrix/sub_3/pfor/MaximumMaximum4loop_body/rotation_matrix/sub_3/pfor/Rank_1:output:02loop_body/rotation_matrix/sub_3/pfor/Rank:output:0*
T0*
_output_shapes
: 
*loop_body/rotation_matrix/sub_3/pfor/ShapeShape*loop_body/rotation_matrix/mul/pfor/Mul:z:0*
T0*
_output_shapes
:¶
(loop_body/rotation_matrix/sub_3/pfor/subSub0loop_body/rotation_matrix/sub_3/pfor/Maximum:z:02loop_body/rotation_matrix/sub_3/pfor/Rank:output:0*
T0*
_output_shapes
: |
2loop_body/rotation_matrix/sub_3/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:Ç
,loop_body/rotation_matrix/sub_3/pfor/ReshapeReshape,loop_body/rotation_matrix/sub_3/pfor/sub:z:0;loop_body/rotation_matrix/sub_3/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:y
/loop_body/rotation_matrix/sub_3/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:Å
)loop_body/rotation_matrix/sub_3/pfor/TileTile8loop_body/rotation_matrix/sub_3/pfor/Tile/input:output:05loop_body/rotation_matrix/sub_3/pfor/Reshape:output:0*
T0*
_output_shapes
: 
8loop_body/rotation_matrix/sub_3/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
:loop_body/rotation_matrix/sub_3/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
:loop_body/rotation_matrix/sub_3/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
2loop_body/rotation_matrix/sub_3/pfor/strided_sliceStridedSlice3loop_body/rotation_matrix/sub_3/pfor/Shape:output:0Aloop_body/rotation_matrix/sub_3/pfor/strided_slice/stack:output:0Cloop_body/rotation_matrix/sub_3/pfor/strided_slice/stack_1:output:0Cloop_body/rotation_matrix/sub_3/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
:loop_body/rotation_matrix/sub_3/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
<loop_body/rotation_matrix/sub_3/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
<loop_body/rotation_matrix/sub_3/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
4loop_body/rotation_matrix/sub_3/pfor/strided_slice_1StridedSlice3loop_body/rotation_matrix/sub_3/pfor/Shape:output:0Cloop_body/rotation_matrix/sub_3/pfor/strided_slice_1/stack:output:0Eloop_body/rotation_matrix/sub_3/pfor/strided_slice_1/stack_1:output:0Eloop_body/rotation_matrix/sub_3/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskr
0loop_body/rotation_matrix/sub_3/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ð
+loop_body/rotation_matrix/sub_3/pfor/concatConcatV2;loop_body/rotation_matrix/sub_3/pfor/strided_slice:output:02loop_body/rotation_matrix/sub_3/pfor/Tile:output:0=loop_body/rotation_matrix/sub_3/pfor/strided_slice_1:output:09loop_body/rotation_matrix/sub_3/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Í
.loop_body/rotation_matrix/sub_3/pfor/Reshape_1Reshape*loop_body/rotation_matrix/mul/pfor/Mul:z:04loop_body/rotation_matrix/sub_3/pfor/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,loop_body/rotation_matrix/sub_3/pfor/Shape_1Shape,loop_body/rotation_matrix/mul_1/pfor/Mul:z:0*
T0*
_output_shapes
:º
*loop_body/rotation_matrix/sub_3/pfor/sub_1Sub0loop_body/rotation_matrix/sub_3/pfor/Maximum:z:04loop_body/rotation_matrix/sub_3/pfor/Rank_1:output:0*
T0*
_output_shapes
: ~
4loop_body/rotation_matrix/sub_3/pfor/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB:Í
.loop_body/rotation_matrix/sub_3/pfor/Reshape_2Reshape.loop_body/rotation_matrix/sub_3/pfor/sub_1:z:0=loop_body/rotation_matrix/sub_3/pfor/Reshape_2/shape:output:0*
T0*
_output_shapes
:{
1loop_body/rotation_matrix/sub_3/pfor/Tile_1/inputConst*
_output_shapes
:*
dtype0*
valueB:Ë
+loop_body/rotation_matrix/sub_3/pfor/Tile_1Tile:loop_body/rotation_matrix/sub_3/pfor/Tile_1/input:output:07loop_body/rotation_matrix/sub_3/pfor/Reshape_2:output:0*
T0*
_output_shapes
: 
:loop_body/rotation_matrix/sub_3/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
<loop_body/rotation_matrix/sub_3/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
<loop_body/rotation_matrix/sub_3/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
4loop_body/rotation_matrix/sub_3/pfor/strided_slice_2StridedSlice5loop_body/rotation_matrix/sub_3/pfor/Shape_1:output:0Cloop_body/rotation_matrix/sub_3/pfor/strided_slice_2/stack:output:0Eloop_body/rotation_matrix/sub_3/pfor/strided_slice_2/stack_1:output:0Eloop_body/rotation_matrix/sub_3/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
:loop_body/rotation_matrix/sub_3/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
<loop_body/rotation_matrix/sub_3/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
<loop_body/rotation_matrix/sub_3/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
4loop_body/rotation_matrix/sub_3/pfor/strided_slice_3StridedSlice5loop_body/rotation_matrix/sub_3/pfor/Shape_1:output:0Cloop_body/rotation_matrix/sub_3/pfor/strided_slice_3/stack:output:0Eloop_body/rotation_matrix/sub_3/pfor/strided_slice_3/stack_1:output:0Eloop_body/rotation_matrix/sub_3/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskt
2loop_body/rotation_matrix/sub_3/pfor/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ø
-loop_body/rotation_matrix/sub_3/pfor/concat_1ConcatV2=loop_body/rotation_matrix/sub_3/pfor/strided_slice_2:output:04loop_body/rotation_matrix/sub_3/pfor/Tile_1:output:0=loop_body/rotation_matrix/sub_3/pfor/strided_slice_3:output:0;loop_body/rotation_matrix/sub_3/pfor/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ñ
.loop_body/rotation_matrix/sub_3/pfor/Reshape_3Reshape,loop_body/rotation_matrix/mul_1/pfor/Mul:z:06loop_body/rotation_matrix/sub_3/pfor/concat_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ
*loop_body/rotation_matrix/sub_3/pfor/Sub_2Sub7loop_body/rotation_matrix/sub_3/pfor/Reshape_1:output:07loop_body/rotation_matrix/sub_3/pfor/Reshape_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
)loop_body/rotation_matrix/sub_4/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B : l
*loop_body/rotation_matrix/sub_4/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :»
(loop_body/rotation_matrix/sub_4/pfor/addAddV22loop_body/rotation_matrix/sub_4/pfor/Rank:output:03loop_body/rotation_matrix/sub_4/pfor/add/y:output:0*
T0*
_output_shapes
: m
+loop_body/rotation_matrix/sub_4/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :¼
,loop_body/rotation_matrix/sub_4/pfor/MaximumMaximum4loop_body/rotation_matrix/sub_4/pfor/Rank_1:output:0,loop_body/rotation_matrix/sub_4/pfor/add:z:0*
T0*
_output_shapes
: 
*loop_body/rotation_matrix/sub_4/pfor/ShapeShape.loop_body/rotation_matrix/sub_3/pfor/Sub_2:z:0*
T0*
_output_shapes
:¸
(loop_body/rotation_matrix/sub_4/pfor/subSub0loop_body/rotation_matrix/sub_4/pfor/Maximum:z:04loop_body/rotation_matrix/sub_4/pfor/Rank_1:output:0*
T0*
_output_shapes
: |
2loop_body/rotation_matrix/sub_4/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:Ç
,loop_body/rotation_matrix/sub_4/pfor/ReshapeReshape,loop_body/rotation_matrix/sub_4/pfor/sub:z:0;loop_body/rotation_matrix/sub_4/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:y
/loop_body/rotation_matrix/sub_4/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:Å
)loop_body/rotation_matrix/sub_4/pfor/TileTile8loop_body/rotation_matrix/sub_4/pfor/Tile/input:output:05loop_body/rotation_matrix/sub_4/pfor/Reshape:output:0*
T0*
_output_shapes
: 
8loop_body/rotation_matrix/sub_4/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
:loop_body/rotation_matrix/sub_4/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
:loop_body/rotation_matrix/sub_4/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
2loop_body/rotation_matrix/sub_4/pfor/strided_sliceStridedSlice3loop_body/rotation_matrix/sub_4/pfor/Shape:output:0Aloop_body/rotation_matrix/sub_4/pfor/strided_slice/stack:output:0Cloop_body/rotation_matrix/sub_4/pfor/strided_slice/stack_1:output:0Cloop_body/rotation_matrix/sub_4/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
:loop_body/rotation_matrix/sub_4/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
<loop_body/rotation_matrix/sub_4/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
<loop_body/rotation_matrix/sub_4/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
4loop_body/rotation_matrix/sub_4/pfor/strided_slice_1StridedSlice3loop_body/rotation_matrix/sub_4/pfor/Shape:output:0Cloop_body/rotation_matrix/sub_4/pfor/strided_slice_1/stack:output:0Eloop_body/rotation_matrix/sub_4/pfor/strided_slice_1/stack_1:output:0Eloop_body/rotation_matrix/sub_4/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskr
0loop_body/rotation_matrix/sub_4/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ð
+loop_body/rotation_matrix/sub_4/pfor/concatConcatV2;loop_body/rotation_matrix/sub_4/pfor/strided_slice:output:02loop_body/rotation_matrix/sub_4/pfor/Tile:output:0=loop_body/rotation_matrix/sub_4/pfor/strided_slice_1:output:09loop_body/rotation_matrix/sub_4/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Ñ
.loop_body/rotation_matrix/sub_4/pfor/Reshape_1Reshape.loop_body/rotation_matrix/sub_3/pfor/Sub_2:z:04loop_body/rotation_matrix/sub_4/pfor/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
*loop_body/rotation_matrix/sub_4/pfor/Sub_1Sub!loop_body/rotation_matrix/sub:z:07loop_body/rotation_matrix/sub_4/pfor/Reshape_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
+loop_body/rotation_matrix/truediv/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :o
-loop_body/rotation_matrix/truediv/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : n
,loop_body/rotation_matrix/truediv/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ã
*loop_body/rotation_matrix/truediv/pfor/addAddV26loop_body/rotation_matrix/truediv/pfor/Rank_1:output:05loop_body/rotation_matrix/truediv/pfor/add/y:output:0*
T0*
_output_shapes
: À
.loop_body/rotation_matrix/truediv/pfor/MaximumMaximum.loop_body/rotation_matrix/truediv/pfor/add:z:04loop_body/rotation_matrix/truediv/pfor/Rank:output:0*
T0*
_output_shapes
: 
,loop_body/rotation_matrix/truediv/pfor/ShapeShape.loop_body/rotation_matrix/sub_4/pfor/Sub_1:z:0*
T0*
_output_shapes
:¼
*loop_body/rotation_matrix/truediv/pfor/subSub2loop_body/rotation_matrix/truediv/pfor/Maximum:z:04loop_body/rotation_matrix/truediv/pfor/Rank:output:0*
T0*
_output_shapes
: ~
4loop_body/rotation_matrix/truediv/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:Í
.loop_body/rotation_matrix/truediv/pfor/ReshapeReshape.loop_body/rotation_matrix/truediv/pfor/sub:z:0=loop_body/rotation_matrix/truediv/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:{
1loop_body/rotation_matrix/truediv/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:Ë
+loop_body/rotation_matrix/truediv/pfor/TileTile:loop_body/rotation_matrix/truediv/pfor/Tile/input:output:07loop_body/rotation_matrix/truediv/pfor/Reshape:output:0*
T0*
_output_shapes
: 
:loop_body/rotation_matrix/truediv/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
<loop_body/rotation_matrix/truediv/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
<loop_body/rotation_matrix/truediv/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
4loop_body/rotation_matrix/truediv/pfor/strided_sliceStridedSlice5loop_body/rotation_matrix/truediv/pfor/Shape:output:0Cloop_body/rotation_matrix/truediv/pfor/strided_slice/stack:output:0Eloop_body/rotation_matrix/truediv/pfor/strided_slice/stack_1:output:0Eloop_body/rotation_matrix/truediv/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
<loop_body/rotation_matrix/truediv/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
>loop_body/rotation_matrix/truediv/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
>loop_body/rotation_matrix/truediv/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
6loop_body/rotation_matrix/truediv/pfor/strided_slice_1StridedSlice5loop_body/rotation_matrix/truediv/pfor/Shape:output:0Eloop_body/rotation_matrix/truediv/pfor/strided_slice_1/stack:output:0Gloop_body/rotation_matrix/truediv/pfor/strided_slice_1/stack_1:output:0Gloop_body/rotation_matrix/truediv/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskt
2loop_body/rotation_matrix/truediv/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ú
-loop_body/rotation_matrix/truediv/pfor/concatConcatV2=loop_body/rotation_matrix/truediv/pfor/strided_slice:output:04loop_body/rotation_matrix/truediv/pfor/Tile:output:0?loop_body/rotation_matrix/truediv/pfor/strided_slice_1:output:0;loop_body/rotation_matrix/truediv/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Õ
0loop_body/rotation_matrix/truediv/pfor/Reshape_1Reshape.loop_body/rotation_matrix/sub_4/pfor/Sub_1:z:06loop_body/rotation_matrix/truediv/pfor/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ
.loop_body/rotation_matrix/truediv/pfor/RealDivRealDiv9loop_body/rotation_matrix/truediv/pfor/Reshape_1:output:0,loop_body/rotation_matrix/truediv/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
>loop_body/rotation_matrix/strided_slice_3/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: |
:loop_body/rotation_matrix/strided_slice_3/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ·
5loop_body/rotation_matrix/strided_slice_3/pfor/concatConcatV2Gloop_body/rotation_matrix/strided_slice_3/pfor/concat/values_0:output:08loop_body/rotation_matrix/strided_slice_3/stack:output:0Cloop_body/rotation_matrix/strided_slice_3/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
@loop_body/rotation_matrix/strided_slice_3/pfor/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB: ~
<loop_body/rotation_matrix/strided_slice_3/pfor/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
7loop_body/rotation_matrix/strided_slice_3/pfor/concat_1ConcatV2Iloop_body/rotation_matrix/strided_slice_3/pfor/concat_1/values_0:output:0:loop_body/rotation_matrix/strided_slice_3/stack_1:output:0Eloop_body/rotation_matrix/strided_slice_3/pfor/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
@loop_body/rotation_matrix/strided_slice_3/pfor/concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB:~
<loop_body/rotation_matrix/strided_slice_3/pfor/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
7loop_body/rotation_matrix/strided_slice_3/pfor/concat_2ConcatV2Iloop_body/rotation_matrix/strided_slice_3/pfor/concat_2/values_0:output:0:loop_body/rotation_matrix/strided_slice_3/stack_2:output:0Eloop_body/rotation_matrix/strided_slice_3/pfor/concat_2/axis:output:0*
N*
T0*
_output_shapes
:½
;loop_body/rotation_matrix/strided_slice_3/pfor/StridedSliceStridedSlice2loop_body/rotation_matrix/truediv/pfor/RealDiv:z:0>loop_body/rotation_matrix/strided_slice_3/pfor/concat:output:0@loop_body/rotation_matrix/strided_slice_3/pfor/concat_1:output:0@loop_body/rotation_matrix/strided_slice_3/pfor/concat_2:output:0*
Index0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask
(loop_body/rotation_matrix/Sin_2/pfor/SinSin)loop_body/stateful_uniform/pfor/AddV2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
>loop_body/rotation_matrix/strided_slice_2/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: |
:loop_body/rotation_matrix/strided_slice_2/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ·
5loop_body/rotation_matrix/strided_slice_2/pfor/concatConcatV2Gloop_body/rotation_matrix/strided_slice_2/pfor/concat/values_0:output:08loop_body/rotation_matrix/strided_slice_2/stack:output:0Cloop_body/rotation_matrix/strided_slice_2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
@loop_body/rotation_matrix/strided_slice_2/pfor/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB: ~
<loop_body/rotation_matrix/strided_slice_2/pfor/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
7loop_body/rotation_matrix/strided_slice_2/pfor/concat_1ConcatV2Iloop_body/rotation_matrix/strided_slice_2/pfor/concat_1/values_0:output:0:loop_body/rotation_matrix/strided_slice_2/stack_1:output:0Eloop_body/rotation_matrix/strided_slice_2/pfor/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
@loop_body/rotation_matrix/strided_slice_2/pfor/concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB:~
<loop_body/rotation_matrix/strided_slice_2/pfor/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
7loop_body/rotation_matrix/strided_slice_2/pfor/concat_2ConcatV2Iloop_body/rotation_matrix/strided_slice_2/pfor/concat_2/values_0:output:0:loop_body/rotation_matrix/strided_slice_2/stack_2:output:0Eloop_body/rotation_matrix/strided_slice_2/pfor/concat_2/axis:output:0*
N*
T0*
_output_shapes
:·
;loop_body/rotation_matrix/strided_slice_2/pfor/StridedSliceStridedSlice,loop_body/rotation_matrix/Sin_2/pfor/Sin:y:0>loop_body/rotation_matrix/strided_slice_2/pfor/concat:output:0@loop_body/rotation_matrix/strided_slice_2/pfor/concat_1:output:0@loop_body/rotation_matrix/strided_slice_2/pfor/concat_2:output:0*
Index0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask©
&loop_body/rotation_matrix/Neg/pfor/NegNegDloop_body/rotation_matrix/strided_slice_2/pfor/StridedSlice:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(loop_body/rotation_matrix/Cos_2/pfor/CosCos)loop_body/stateful_uniform/pfor/AddV2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
>loop_body/rotation_matrix/strided_slice_1/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: |
:loop_body/rotation_matrix/strided_slice_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ·
5loop_body/rotation_matrix/strided_slice_1/pfor/concatConcatV2Gloop_body/rotation_matrix/strided_slice_1/pfor/concat/values_0:output:08loop_body/rotation_matrix/strided_slice_1/stack:output:0Cloop_body/rotation_matrix/strided_slice_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
@loop_body/rotation_matrix/strided_slice_1/pfor/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB: ~
<loop_body/rotation_matrix/strided_slice_1/pfor/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
7loop_body/rotation_matrix/strided_slice_1/pfor/concat_1ConcatV2Iloop_body/rotation_matrix/strided_slice_1/pfor/concat_1/values_0:output:0:loop_body/rotation_matrix/strided_slice_1/stack_1:output:0Eloop_body/rotation_matrix/strided_slice_1/pfor/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
@loop_body/rotation_matrix/strided_slice_1/pfor/concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB:~
<loop_body/rotation_matrix/strided_slice_1/pfor/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
7loop_body/rotation_matrix/strided_slice_1/pfor/concat_2ConcatV2Iloop_body/rotation_matrix/strided_slice_1/pfor/concat_2/values_0:output:0:loop_body/rotation_matrix/strided_slice_1/stack_2:output:0Eloop_body/rotation_matrix/strided_slice_1/pfor/concat_2/axis:output:0*
N*
T0*
_output_shapes
:·
;loop_body/rotation_matrix/strided_slice_1/pfor/StridedSliceStridedSlice,loop_body/rotation_matrix/Cos_2/pfor/Cos:y:0>loop_body/rotation_matrix/strided_slice_1/pfor/concat:output:0@loop_body/rotation_matrix/strided_slice_1/pfor/concat_1:output:0@loop_body/rotation_matrix/strided_slice_1/pfor/concat_2:output:0*
Index0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask|
+loop_body/rotation_matrix/concat/pfor/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      
Eloop_body/rotation_matrix/concat/pfor/ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:w
5loop_body/rotation_matrix/concat/pfor/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :ì
/loop_body/rotation_matrix/concat/pfor/ones_likeFillNloop_body/rotation_matrix/concat/pfor/ones_like/Shape/shape_as_tensor:output:0>loop_body/rotation_matrix/concat/pfor/ones_like/Const:output:0*
T0*
_output_shapes
:
3loop_body/rotation_matrix/concat/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿÕ
-loop_body/rotation_matrix/concat/pfor/ReshapeReshape8loop_body/rotation_matrix/concat/pfor/ones_like:output:0<loop_body/rotation_matrix/concat/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:
5loop_body/rotation_matrix/concat/pfor/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ¶
/loop_body/rotation_matrix/concat/pfor/Reshape_1Reshapepfor/Reshape:output:0>loop_body/rotation_matrix/concat/pfor/Reshape_1/shape:output:0*
T0*
_output_shapes
:s
1loop_body/rotation_matrix/concat/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
,loop_body/rotation_matrix/concat/pfor/concatConcatV28loop_body/rotation_matrix/concat/pfor/Reshape_1:output:06loop_body/rotation_matrix/concat/pfor/Reshape:output:0:loop_body/rotation_matrix/concat/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:v
4loop_body/rotation_matrix/concat/pfor/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : Ô
0loop_body/rotation_matrix/concat/pfor/ExpandDims
ExpandDims(loop_body/rotation_matrix/zeros:output:0=loop_body/rotation_matrix/concat/pfor/ExpandDims/dim:output:0*
T0*"
_output_shapes
:Ú
*loop_body/rotation_matrix/concat/pfor/TileTile9loop_body/rotation_matrix/concat/pfor/ExpandDims:output:05loop_body/rotation_matrix/concat/pfor/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
4loop_body/rotation_matrix/concat/pfor/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : Ò
2loop_body/rotation_matrix/concat/pfor/GreaterEqualGreaterEqual.loop_body/rotation_matrix/concat/axis:output:0=loop_body/rotation_matrix/concat/pfor/GreaterEqual/y:output:0*
T0*
_output_shapes
: 
*loop_body/rotation_matrix/concat/pfor/CastCast6loop_body/rotation_matrix/concat/pfor/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
: ³
)loop_body/rotation_matrix/concat/pfor/addAddV2.loop_body/rotation_matrix/concat/axis:output:0.loop_body/rotation_matrix/concat/pfor/Cast:y:0*
T0*
_output_shapes
: ç
.loop_body/rotation_matrix/concat/pfor/concat_1ConcatV2Dloop_body/rotation_matrix/strided_slice_1/pfor/StridedSlice:output:0*loop_body/rotation_matrix/Neg/pfor/Neg:y:0Dloop_body/rotation_matrix/strided_slice_3/pfor/StridedSlice:output:0Dloop_body/rotation_matrix/strided_slice_4/pfor/StridedSlice:output:0Dloop_body/rotation_matrix/strided_slice_5/pfor/StridedSlice:output:0Dloop_body/rotation_matrix/strided_slice_6/pfor/StridedSlice:output:03loop_body/rotation_matrix/concat/pfor/Tile:output:0-loop_body/rotation_matrix/concat/pfor/add:z:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
loop_body/SelectV2/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B : _
loop_body/SelectV2/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :
loop_body/SelectV2/pfor/addAddV2%loop_body/SelectV2/pfor/Rank:output:0&loop_body/SelectV2/pfor/add/y:output:0*
T0*
_output_shapes
: `
loop_body/SelectV2/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :`
loop_body/SelectV2/pfor/Rank_2Const*
_output_shapes
: *
dtype0*
value	B : a
loop_body/SelectV2/pfor/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
loop_body/SelectV2/pfor/add_1AddV2'loop_body/SelectV2/pfor/Rank_2:output:0(loop_body/SelectV2/pfor/add_1/y:output:0*
T0*
_output_shapes
: 
loop_body/SelectV2/pfor/MaximumMaximum'loop_body/SelectV2/pfor/Rank_1:output:0loop_body/SelectV2/pfor/add:z:0*
T0*
_output_shapes
: 
!loop_body/SelectV2/pfor/Maximum_1Maximum!loop_body/SelectV2/pfor/add_1:z:0#loop_body/SelectV2/pfor/Maximum:z:0*
T0*
_output_shapes
: `
loop_body/SelectV2/pfor/ShapeShapepfor/range:output:0*
T0*
_output_shapes
:
loop_body/SelectV2/pfor/subSub%loop_body/SelectV2/pfor/Maximum_1:z:0'loop_body/SelectV2/pfor/Rank_1:output:0*
T0*
_output_shapes
: o
%loop_body/SelectV2/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB: 
loop_body/SelectV2/pfor/ReshapeReshapeloop_body/SelectV2/pfor/sub:z:0.loop_body/SelectV2/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:l
"loop_body/SelectV2/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:
loop_body/SelectV2/pfor/TileTile+loop_body/SelectV2/pfor/Tile/input:output:0(loop_body/SelectV2/pfor/Reshape:output:0*
T0*
_output_shapes
: u
+loop_body/SelectV2/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-loop_body/SelectV2/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-loop_body/SelectV2/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ç
%loop_body/SelectV2/pfor/strided_sliceStridedSlice&loop_body/SelectV2/pfor/Shape:output:04loop_body/SelectV2/pfor/strided_slice/stack:output:06loop_body/SelectV2/pfor/strided_slice/stack_1:output:06loop_body/SelectV2/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskw
-loop_body/SelectV2/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:y
/loop_body/SelectV2/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: y
/loop_body/SelectV2/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ë
'loop_body/SelectV2/pfor/strided_slice_1StridedSlice&loop_body/SelectV2/pfor/Shape:output:06loop_body/SelectV2/pfor/strided_slice_1/stack:output:08loop_body/SelectV2/pfor/strided_slice_1/stack_1:output:08loop_body/SelectV2/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maske
#loop_body/SelectV2/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
loop_body/SelectV2/pfor/concatConcatV2.loop_body/SelectV2/pfor/strided_slice:output:0%loop_body/SelectV2/pfor/Tile:output:00loop_body/SelectV2/pfor/strided_slice_1:output:0,loop_body/SelectV2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
!loop_body/SelectV2/pfor/Reshape_1Reshapepfor/range:output:0'loop_body/SelectV2/pfor/concat:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
 loop_body/SelectV2/pfor/SelectV2SelectV2loop_body/Greater:z:0*loop_body/SelectV2/pfor/Reshape_1:output:0loop_body/SelectV2/e:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
%loop_body/GatherV2/pfor/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : î
 loop_body/GatherV2/pfor/GatherV2GatherV2inputs)loop_body/SelectV2/pfor/SelectV2:output:0.loop_body/GatherV2/pfor/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
(loop_body/ExpandDims/pfor/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ­
&loop_body/ExpandDims/pfor/GreaterEqualGreaterEqual!loop_body/ExpandDims/dim:output:01loop_body/ExpandDims/pfor/GreaterEqual/y:output:0*
T0*
_output_shapes
: 
loop_body/ExpandDims/pfor/CastCast*loop_body/ExpandDims/pfor/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
: 
loop_body/ExpandDims/pfor/addAddV2!loop_body/ExpandDims/dim:output:0"loop_body/ExpandDims/pfor/Cast:y:0*
T0*
_output_shapes
: À
$loop_body/ExpandDims/pfor/ExpandDims
ExpandDims)loop_body/GatherV2/pfor/GatherV2:output:0!loop_body/ExpandDims/pfor/add:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿ
Gloop_body/transform/ImageProjectiveTransformV3/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Iloop_body/transform/ImageProjectiveTransformV3/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Iloop_body/transform/ImageProjectiveTransformV3/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¨
Aloop_body/transform/ImageProjectiveTransformV3/pfor/strided_sliceStridedSlicepfor/Reshape:output:0Ploop_body/transform/ImageProjectiveTransformV3/pfor/strided_slice/stack:output:0Rloop_body/transform/ImageProjectiveTransformV3/pfor/strided_slice/stack_1:output:0Rloop_body/transform/ImageProjectiveTransformV3/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Oloop_body/transform/ImageProjectiveTransformV3/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÎ
Aloop_body/transform/ImageProjectiveTransformV3/pfor/TensorArrayV2TensorListReserveXloop_body/transform/ImageProjectiveTransformV3/pfor/TensorArrayV2/element_shape:output:0Jloop_body/transform/ImageProjectiveTransformV3/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ{
9loop_body/transform/ImageProjectiveTransformV3/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
Lloop_body/transform/ImageProjectiveTransformV3/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Floop_body/transform/ImageProjectiveTransformV3/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : þ
9loop_body/transform/ImageProjectiveTransformV3/pfor/whileStatelessWhileOloop_body/transform/ImageProjectiveTransformV3/pfor/while/loop_counter:output:0Uloop_body/transform/ImageProjectiveTransformV3/pfor/while/maximum_iterations:output:0Bloop_body/transform/ImageProjectiveTransformV3/pfor/Const:output:0Jloop_body/transform/ImageProjectiveTransformV3/pfor/TensorArrayV2:handle:0Jloop_body/transform/ImageProjectiveTransformV3/pfor/strided_slice:output:0-loop_body/ExpandDims/pfor/ExpandDims:output:07loop_body/rotation_matrix/concat/pfor/concat_1:output:0*loop_body/transform/strided_slice:output:0'loop_body/transform/fill_value:output:0*
T
2	*
_lower_using_switch_merge(*
_num_original_outputs	*^
_output_shapesL
J: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: * 
_read_only_resource_inputs
 *
_stateful_parallelism( *O
bodyGRE
Cloop_body_transform_ImageProjectiveTransformV3_pfor_while_body_6355*O
condGRE
Cloop_body_transform_ImageProjectiveTransformV3_pfor_while_cond_6354*]
output_shapesL
J: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: ~
;loop_body/transform/ImageProjectiveTransformV3/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 ±
Tloop_body/transform/ImageProjectiveTransformV3/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*)
value B"ÿÿÿÿ   ÿÿÿÿÿÿÿÿ   ²
Floop_body/transform/ImageProjectiveTransformV3/pfor/TensorListConcatV2TensorListConcatV2Bloop_body/transform/ImageProjectiveTransformV3/pfor/while:output:3]loop_body/transform/ImageProjectiveTransformV3/pfor/TensorListConcatV2/element_shape:output:0Dloop_body/transform/ImageProjectiveTransformV3/pfor/Const_1:output:0*D
_output_shapes2
0:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0Í
loop_body/Squeeze/pfor/SqueezeSqueezeOloop_body/transform/ImageProjectiveTransformV3/pfor/TensorListConcatV2:tensor:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

IdentityIdentity'loop_body/Squeeze/pfor/Squeeze:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©
NoOpNoOp*^loop_body/stateful_uniform/RngReadAndSkip5^loop_body/stateful_uniform/RngReadAndSkip/pfor/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: 2V
)loop_body/stateful_uniform/RngReadAndSkip)loop_body/stateful_uniform/RngReadAndSkip2l
4loop_body/stateful_uniform/RngReadAndSkip/pfor/while4loop_body/stateful_uniform/RngReadAndSkip/pfor/while:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±¿

I__inference_random_rotation_layer_call_and_return_conditional_losses_8793

inputs@
2loop_body_stateful_uniform_rngreadandskip_resource:	
identity¢)loop_body/stateful_uniform/RngReadAndSkip¢4loop_body/stateful_uniform/RngReadAndSkip/pfor/while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskY
Rank/packedPackstrided_slice:output:0*
N*
T0*
_output_shapes
:F
RankConst*
_output_shapes
: *
dtype0*
value	B :M
range/startConst*
_output_shapes
: *
dtype0*
value	B : M
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :e
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*
_output_shapes
:W
	Max/inputPackstrided_slice:output:0*
N*
T0*
_output_shapes
:O
MaxMaxMax/input:output:0range:output:0*
T0*
_output_shapes
: h
&loop_body/PlaceholderWithDefault/inputConst*
_output_shapes
: *
dtype0*
value	B : 
 loop_body/PlaceholderWithDefaultPlaceholderWithDefault/loop_body/PlaceholderWithDefault/input:output:0*
_output_shapes
: *
dtype0*
shape: E
loop_body/ShapeShapeinputs*
T0*
_output_shapes
:g
loop_body/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: i
loop_body/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
loop_body/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
loop_body/strided_sliceStridedSliceloop_body/Shape:output:0&loop_body/strided_slice/stack:output:0(loop_body/strided_slice/stack_1:output:0(loop_body/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskU
loop_body/Greater/yConst*
_output_shapes
: *
dtype0*
value	B :}
loop_body/GreaterGreater loop_body/strided_slice:output:0loop_body/Greater/y:output:0*
T0*
_output_shapes
: V
loop_body/SelectV2/eConst*
_output_shapes
: *
dtype0*
value	B :  
loop_body/SelectV2SelectV2loop_body/Greater:z:0)loop_body/PlaceholderWithDefault:output:0loop_body/SelectV2/e:output:0*
T0*
_output_shapes
: Y
loop_body/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ·
loop_body/GatherV2GatherV2inputsloop_body/SelectV2:output:0 loop_body/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*$
_output_shapes
:j
 loop_body/stateful_uniform/shapeConst*
_output_shapes
:*
dtype0*
valueB:c
loop_body/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|Ù ¿c
loop_body/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|Ù ?j
 loop_body/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
loop_body/stateful_uniform/ProdProd)loop_body/stateful_uniform/shape:output:0)loop_body/stateful_uniform/Const:output:0*
T0*
_output_shapes
: c
!loop_body/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :
!loop_body/stateful_uniform/Cast_1Cast(loop_body/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: Þ
)loop_body/stateful_uniform/RngReadAndSkipRngReadAndSkip2loop_body_stateful_uniform_rngreadandskip_resource*loop_body/stateful_uniform/Cast/x:output:0%loop_body/stateful_uniform/Cast_1:y:0*
_output_shapes
:x
.loop_body/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: z
0loop_body/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:z
0loop_body/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Þ
(loop_body/stateful_uniform/strided_sliceStridedSlice1loop_body/stateful_uniform/RngReadAndSkip:value:07loop_body/stateful_uniform/strided_slice/stack:output:09loop_body/stateful_uniform/strided_slice/stack_1:output:09loop_body/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask
"loop_body/stateful_uniform/BitcastBitcast1loop_body/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type0z
0loop_body/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:|
2loop_body/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:|
2loop_body/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ô
*loop_body/stateful_uniform/strided_slice_1StridedSlice1loop_body/stateful_uniform/RngReadAndSkip:value:09loop_body/stateful_uniform/strided_slice_1/stack:output:0;loop_body/stateful_uniform/strided_slice_1/stack_1:output:0;loop_body/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:
$loop_body/stateful_uniform/Bitcast_1Bitcast3loop_body/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type0y
7loop_body/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :´
3loop_body/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2)loop_body/stateful_uniform/shape:output:0-loop_body/stateful_uniform/Bitcast_1:output:0+loop_body/stateful_uniform/Bitcast:output:0@loop_body/stateful_uniform/StatelessRandomUniformV2/alg:output:0*
_output_shapes
:
loop_body/stateful_uniform/subSub'loop_body/stateful_uniform/max:output:0'loop_body/stateful_uniform/min:output:0*
T0*
_output_shapes
: ¬
loop_body/stateful_uniform/mulMul<loop_body/stateful_uniform/StatelessRandomUniformV2:output:0"loop_body/stateful_uniform/sub:z:0*
T0*
_output_shapes
:
loop_body/stateful_uniformAddV2"loop_body/stateful_uniform/mul:z:0'loop_body/stateful_uniform/min:output:0*
T0*
_output_shapes
:Z
loop_body/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : 
loop_body/ExpandDims
ExpandDimsloop_body/GatherV2:output:0!loop_body/ExpandDims/dim:output:0*
T0*(
_output_shapes
:j
loop_body/Shape_1Const*
_output_shapes
:*
dtype0*%
valueB"            r
loop_body/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
ýÿÿÿÿÿÿÿÿt
!loop_body/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿk
!loop_body/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
loop_body/strided_slice_1StridedSliceloop_body/Shape_1:output:0(loop_body/strided_slice_1/stack:output:0*loop_body/strided_slice_1/stack_1:output:0*loop_body/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskj
loop_body/CastCast"loop_body/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: r
loop_body/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:
þÿÿÿÿÿÿÿÿt
!loop_body/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿk
!loop_body/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
loop_body/strided_slice_2StridedSliceloop_body/Shape_1:output:0(loop_body/strided_slice_2/stack:output:0*loop_body/strided_slice_2/stack_1:output:0*loop_body/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskl
loop_body/Cast_1Cast"loop_body/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: d
loop_body/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
loop_body/rotation_matrix/subSubloop_body/Cast_1:y:0(loop_body/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: i
loop_body/rotation_matrix/CosCosloop_body/stateful_uniform:z:0*
T0*
_output_shapes
:f
!loop_body/rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
loop_body/rotation_matrix/sub_1Subloop_body/Cast_1:y:0*loop_body/rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 
loop_body/rotation_matrix/mulMul!loop_body/rotation_matrix/Cos:y:0#loop_body/rotation_matrix/sub_1:z:0*
T0*
_output_shapes
:i
loop_body/rotation_matrix/SinSinloop_body/stateful_uniform:z:0*
T0*
_output_shapes
:f
!loop_body/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
loop_body/rotation_matrix/sub_2Subloop_body/Cast:y:0*loop_body/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 
loop_body/rotation_matrix/mul_1Mul!loop_body/rotation_matrix/Sin:y:0#loop_body/rotation_matrix/sub_2:z:0*
T0*
_output_shapes
:
loop_body/rotation_matrix/sub_3Sub!loop_body/rotation_matrix/mul:z:0#loop_body/rotation_matrix/mul_1:z:0*
T0*
_output_shapes
:
loop_body/rotation_matrix/sub_4Sub!loop_body/rotation_matrix/sub:z:0#loop_body/rotation_matrix/sub_3:z:0*
T0*
_output_shapes
:h
#loop_body/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @¤
!loop_body/rotation_matrix/truedivRealDiv#loop_body/rotation_matrix/sub_4:z:0,loop_body/rotation_matrix/truediv/y:output:0*
T0*
_output_shapes
:f
!loop_body/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
loop_body/rotation_matrix/sub_5Subloop_body/Cast:y:0*loop_body/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: k
loop_body/rotation_matrix/Sin_1Sinloop_body/stateful_uniform:z:0*
T0*
_output_shapes
:f
!loop_body/rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
loop_body/rotation_matrix/sub_6Subloop_body/Cast_1:y:0*loop_body/rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 
loop_body/rotation_matrix/mul_2Mul#loop_body/rotation_matrix/Sin_1:y:0#loop_body/rotation_matrix/sub_6:z:0*
T0*
_output_shapes
:k
loop_body/rotation_matrix/Cos_1Cosloop_body/stateful_uniform:z:0*
T0*
_output_shapes
:f
!loop_body/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
loop_body/rotation_matrix/sub_7Subloop_body/Cast:y:0*loop_body/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 
loop_body/rotation_matrix/mul_3Mul#loop_body/rotation_matrix/Cos_1:y:0#loop_body/rotation_matrix/sub_7:z:0*
T0*
_output_shapes
:
loop_body/rotation_matrix/addAddV2#loop_body/rotation_matrix/mul_2:z:0#loop_body/rotation_matrix/mul_3:z:0*
T0*
_output_shapes
:
loop_body/rotation_matrix/sub_8Sub#loop_body/rotation_matrix/sub_5:z:0!loop_body/rotation_matrix/add:z:0*
T0*
_output_shapes
:j
%loop_body/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @¨
#loop_body/rotation_matrix/truediv_1RealDiv#loop_body/rotation_matrix/sub_8:z:0.loop_body/rotation_matrix/truediv_1/y:output:0*
T0*
_output_shapes
:i
loop_body/rotation_matrix/ShapeConst*
_output_shapes
:*
dtype0*
valueB:w
-loop_body/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: y
/loop_body/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:y
/loop_body/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ó
'loop_body/rotation_matrix/strided_sliceStridedSlice(loop_body/rotation_matrix/Shape:output:06loop_body/rotation_matrix/strided_slice/stack:output:08loop_body/rotation_matrix/strided_slice/stack_1:output:08loop_body/rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskk
loop_body/rotation_matrix/Cos_2Cosloop_body/stateful_uniform:z:0*
T0*
_output_shapes
:
/loop_body/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        
1loop_body/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
1loop_body/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ý
)loop_body/rotation_matrix/strided_slice_1StridedSlice#loop_body/rotation_matrix/Cos_2:y:08loop_body/rotation_matrix/strided_slice_1/stack:output:0:loop_body/rotation_matrix/strided_slice_1/stack_1:output:0:loop_body/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask*
new_axis_maskk
loop_body/rotation_matrix/Sin_2Sinloop_body/stateful_uniform:z:0*
T0*
_output_shapes
:
/loop_body/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        
1loop_body/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
1loop_body/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ý
)loop_body/rotation_matrix/strided_slice_2StridedSlice#loop_body/rotation_matrix/Sin_2:y:08loop_body/rotation_matrix/strided_slice_2/stack:output:0:loop_body/rotation_matrix/strided_slice_2/stack_1:output:0:loop_body/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask*
new_axis_mask
loop_body/rotation_matrix/NegNeg2loop_body/rotation_matrix/strided_slice_2:output:0*
T0*
_output_shapes

:
/loop_body/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        
1loop_body/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
1loop_body/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ÿ
)loop_body/rotation_matrix/strided_slice_3StridedSlice%loop_body/rotation_matrix/truediv:z:08loop_body/rotation_matrix/strided_slice_3/stack:output:0:loop_body/rotation_matrix/strided_slice_3/stack_1:output:0:loop_body/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask*
new_axis_maskk
loop_body/rotation_matrix/Sin_3Sinloop_body/stateful_uniform:z:0*
T0*
_output_shapes
:
/loop_body/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        
1loop_body/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
1loop_body/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ý
)loop_body/rotation_matrix/strided_slice_4StridedSlice#loop_body/rotation_matrix/Sin_3:y:08loop_body/rotation_matrix/strided_slice_4/stack:output:0:loop_body/rotation_matrix/strided_slice_4/stack_1:output:0:loop_body/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask*
new_axis_maskk
loop_body/rotation_matrix/Cos_3Cosloop_body/stateful_uniform:z:0*
T0*
_output_shapes
:
/loop_body/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        
1loop_body/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
1loop_body/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ý
)loop_body/rotation_matrix/strided_slice_5StridedSlice#loop_body/rotation_matrix/Cos_3:y:08loop_body/rotation_matrix/strided_slice_5/stack:output:0:loop_body/rotation_matrix/strided_slice_5/stack_1:output:0:loop_body/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask*
new_axis_mask
/loop_body/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        
1loop_body/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
1loop_body/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
)loop_body/rotation_matrix/strided_slice_6StridedSlice'loop_body/rotation_matrix/truediv_1:z:08loop_body/rotation_matrix/strided_slice_6/stack:output:0:loop_body/rotation_matrix/strided_slice_6/stack_1:output:0:loop_body/rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask*
new_axis_maskj
(loop_body/rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Á
&loop_body/rotation_matrix/zeros/packedPack0loop_body/rotation_matrix/strided_slice:output:01loop_body/rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:j
%loop_body/rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ±
loop_body/rotation_matrix/zerosFill/loop_body/rotation_matrix/zeros/packed:output:0.loop_body/rotation_matrix/zeros/Const:output:0*
T0*
_output_shapes

:g
%loop_body/rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ß
 loop_body/rotation_matrix/concatConcatV22loop_body/rotation_matrix/strided_slice_1:output:0!loop_body/rotation_matrix/Neg:y:02loop_body/rotation_matrix/strided_slice_3:output:02loop_body/rotation_matrix/strided_slice_4:output:02loop_body/rotation_matrix/strided_slice_5:output:02loop_body/rotation_matrix/strided_slice_6:output:0(loop_body/rotation_matrix/zeros:output:0.loop_body/rotation_matrix/concat/axis:output:0*
N*
T0*
_output_shapes

:r
loop_body/transform/ShapeConst*
_output_shapes
:*
dtype0*%
valueB"            q
'loop_body/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:s
)loop_body/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)loop_body/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¡
!loop_body/transform/strided_sliceStridedSlice"loop_body/transform/Shape:output:00loop_body/transform/strided_slice/stack:output:02loop_body/transform/strided_slice/stack_1:output:02loop_body/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:c
loop_body/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    ×
.loop_body/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3loop_body/ExpandDims:output:0)loop_body/rotation_matrix/concat:output:0*loop_body/transform/strided_slice:output:0'loop_body/transform/fill_value:output:0*(
_output_shapes
:*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR§
loop_body/SqueezeSqueezeCloop_body/transform/ImageProjectiveTransformV3:transformed_images:0*
T0*$
_output_shapes
:*
squeeze_dims
 \
pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:g
pfor/ReshapeReshapeMax:output:0pfor/Reshape/shape:output:0*
T0*
_output_shapes
:R
pfor/range/startConst*
_output_shapes
: *
dtype0*
value	B : R
pfor/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :|

pfor/rangeRangepfor/range/start:output:0Max:output:0pfor/range/delta:output:0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Bloop_body/stateful_uniform/RngReadAndSkip/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Dloop_body/stateful_uniform/RngReadAndSkip/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Dloop_body/stateful_uniform/RngReadAndSkip/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
<loop_body/stateful_uniform/RngReadAndSkip/pfor/strided_sliceStridedSlicepfor/Reshape:output:0Kloop_body/stateful_uniform/RngReadAndSkip/pfor/strided_slice/stack:output:0Mloop_body/stateful_uniform/RngReadAndSkip/pfor/strided_slice/stack_1:output:0Mloop_body/stateful_uniform/RngReadAndSkip/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Jloop_body/stateful_uniform/RngReadAndSkip/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ¿
<loop_body/stateful_uniform/RngReadAndSkip/pfor/TensorArrayV2TensorListReserveSloop_body/stateful_uniform/RngReadAndSkip/pfor/TensorArrayV2/element_shape:output:0Eloop_body/stateful_uniform/RngReadAndSkip/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0	*

shape_type0:éèÐv
4loop_body/stateful_uniform/RngReadAndSkip/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
Gloop_body/stateful_uniform/RngReadAndSkip/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Aloop_body/stateful_uniform/RngReadAndSkip/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Î
4loop_body/stateful_uniform/RngReadAndSkip/pfor/whileWhileJloop_body/stateful_uniform/RngReadAndSkip/pfor/while/loop_counter:output:0Ploop_body/stateful_uniform/RngReadAndSkip/pfor/while/maximum_iterations:output:0=loop_body/stateful_uniform/RngReadAndSkip/pfor/Const:output:0Eloop_body/stateful_uniform/RngReadAndSkip/pfor/TensorArrayV2:handle:0Eloop_body/stateful_uniform/RngReadAndSkip/pfor/strided_slice:output:02loop_body_stateful_uniform_rngreadandskip_resource*loop_body/stateful_uniform/Cast/x:output:0%loop_body/stateful_uniform/Cast_1:y:0*^loop_body/stateful_uniform/RngReadAndSkip*
T

2*
_lower_using_switch_merge(*
_num_original_outputs*$
_output_shapes
: : : : : : : : * 
_read_only_resource_inputs
 *
_stateful_parallelism( *J
bodyBR@
>loop_body_stateful_uniform_RngReadAndSkip_pfor_while_body_8035*J
condBR@
>loop_body_stateful_uniform_RngReadAndSkip_pfor_while_cond_8034*#
output_shapes
: : : : : : : : y
6loop_body/stateful_uniform/RngReadAndSkip/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	  
Oloop_body/stateful_uniform/RngReadAndSkip/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
Aloop_body/stateful_uniform/RngReadAndSkip/pfor/TensorListConcatV2TensorListConcatV2=loop_body/stateful_uniform/RngReadAndSkip/pfor/while:output:3Xloop_body/stateful_uniform/RngReadAndSkip/pfor/TensorListConcatV2/element_shape:output:0?loop_body/stateful_uniform/RngReadAndSkip/pfor/Const_1:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0	*

shape_type0
=loop_body/stateful_uniform/strided_slice/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: {
9loop_body/stateful_uniform/strided_slice/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ³
4loop_body/stateful_uniform/strided_slice/pfor/concatConcatV2Floop_body/stateful_uniform/strided_slice/pfor/concat/values_0:output:07loop_body/stateful_uniform/strided_slice/stack:output:0Bloop_body/stateful_uniform/strided_slice/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
?loop_body/stateful_uniform/strided_slice/pfor/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB: }
;loop_body/stateful_uniform/strided_slice/pfor/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : »
6loop_body/stateful_uniform/strided_slice/pfor/concat_1ConcatV2Hloop_body/stateful_uniform/strided_slice/pfor/concat_1/values_0:output:09loop_body/stateful_uniform/strided_slice/stack_1:output:0Dloop_body/stateful_uniform/strided_slice/pfor/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
?loop_body/stateful_uniform/strided_slice/pfor/concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB:}
;loop_body/stateful_uniform/strided_slice/pfor/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : »
6loop_body/stateful_uniform/strided_slice/pfor/concat_2ConcatV2Hloop_body/stateful_uniform/strided_slice/pfor/concat_2/values_0:output:09loop_body/stateful_uniform/strided_slice/stack_2:output:0Dloop_body/stateful_uniform/strided_slice/pfor/concat_2/axis:output:0*
N*
T0*
_output_shapes
:¸
:loop_body/stateful_uniform/strided_slice/pfor/StridedSliceStridedSliceJloop_body/stateful_uniform/RngReadAndSkip/pfor/TensorListConcatV2:tensor:0=loop_body/stateful_uniform/strided_slice/pfor/concat:output:0?loop_body/stateful_uniform/strided_slice/pfor/concat_1:output:0?loop_body/stateful_uniform/strided_slice/pfor/concat_2:output:0*
Index0*
T0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask
;loop_body/stateful_uniform/Bitcast/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
=loop_body/stateful_uniform/Bitcast/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
=loop_body/stateful_uniform/Bitcast/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ø
5loop_body/stateful_uniform/Bitcast/pfor/strided_sliceStridedSlicepfor/Reshape:output:0Dloop_body/stateful_uniform/Bitcast/pfor/strided_slice/stack:output:0Floop_body/stateful_uniform/Bitcast/pfor/strided_slice/stack_1:output:0Floop_body/stateful_uniform/Bitcast/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Cloop_body/stateful_uniform/Bitcast/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿª
5loop_body/stateful_uniform/Bitcast/pfor/TensorArrayV2TensorListReserveLloop_body/stateful_uniform/Bitcast/pfor/TensorArrayV2/element_shape:output:0>loop_body/stateful_uniform/Bitcast/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÌo
-loop_body/stateful_uniform/Bitcast/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
@loop_body/stateful_uniform/Bitcast/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ|
:loop_body/stateful_uniform/Bitcast/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : É
-loop_body/stateful_uniform/Bitcast/pfor/whileStatelessWhileCloop_body/stateful_uniform/Bitcast/pfor/while/loop_counter:output:0Iloop_body/stateful_uniform/Bitcast/pfor/while/maximum_iterations:output:06loop_body/stateful_uniform/Bitcast/pfor/Const:output:0>loop_body/stateful_uniform/Bitcast/pfor/TensorArrayV2:handle:0>loop_body/stateful_uniform/Bitcast/pfor/strided_slice:output:0Cloop_body/stateful_uniform/strided_slice/pfor/StridedSlice:output:0*
T

2	*
_lower_using_switch_merge(*
_num_original_outputs*1
_output_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *
_stateful_parallelism( *C
body;R9
7loop_body_stateful_uniform_Bitcast_pfor_while_body_8100*C
cond;R9
7loop_body_stateful_uniform_Bitcast_pfor_while_cond_8099*0
output_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿr
/loop_body/stateful_uniform/Bitcast/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 
Hloop_body/stateful_uniform/Bitcast/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ô
:loop_body/stateful_uniform/Bitcast/pfor/TensorListConcatV2TensorListConcatV26loop_body/stateful_uniform/Bitcast/pfor/while:output:3Qloop_body/stateful_uniform/Bitcast/pfor/TensorListConcatV2/element_shape:output:08loop_body/stateful_uniform/Bitcast/pfor/Const_1:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0
?loop_body/stateful_uniform/strided_slice_1/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: }
;loop_body/stateful_uniform/strided_slice_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : »
6loop_body/stateful_uniform/strided_slice_1/pfor/concatConcatV2Hloop_body/stateful_uniform/strided_slice_1/pfor/concat/values_0:output:09loop_body/stateful_uniform/strided_slice_1/stack:output:0Dloop_body/stateful_uniform/strided_slice_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
Aloop_body/stateful_uniform/strided_slice_1/pfor/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB: 
=loop_body/stateful_uniform/strided_slice_1/pfor/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ã
8loop_body/stateful_uniform/strided_slice_1/pfor/concat_1ConcatV2Jloop_body/stateful_uniform/strided_slice_1/pfor/concat_1/values_0:output:0;loop_body/stateful_uniform/strided_slice_1/stack_1:output:0Floop_body/stateful_uniform/strided_slice_1/pfor/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
Aloop_body/stateful_uniform/strided_slice_1/pfor/concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB:
=loop_body/stateful_uniform/strided_slice_1/pfor/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ã
8loop_body/stateful_uniform/strided_slice_1/pfor/concat_2ConcatV2Jloop_body/stateful_uniform/strided_slice_1/pfor/concat_2/values_0:output:0;loop_body/stateful_uniform/strided_slice_1/stack_2:output:0Floop_body/stateful_uniform/strided_slice_1/pfor/concat_2/axis:output:0*
N*
T0*
_output_shapes
:À
<loop_body/stateful_uniform/strided_slice_1/pfor/StridedSliceStridedSliceJloop_body/stateful_uniform/RngReadAndSkip/pfor/TensorListConcatV2:tensor:0?loop_body/stateful_uniform/strided_slice_1/pfor/concat:output:0Aloop_body/stateful_uniform/strided_slice_1/pfor/concat_1:output:0Aloop_body/stateful_uniform/strided_slice_1/pfor/concat_2:output:0*
Index0*
T0	*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask
=loop_body/stateful_uniform/Bitcast_1/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
?loop_body/stateful_uniform/Bitcast_1/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
?loop_body/stateful_uniform/Bitcast_1/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
7loop_body/stateful_uniform/Bitcast_1/pfor/strided_sliceStridedSlicepfor/Reshape:output:0Floop_body/stateful_uniform/Bitcast_1/pfor/strided_slice/stack:output:0Hloop_body/stateful_uniform/Bitcast_1/pfor/strided_slice/stack_1:output:0Hloop_body/stateful_uniform/Bitcast_1/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Eloop_body/stateful_uniform/Bitcast_1/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ°
7loop_body/stateful_uniform/Bitcast_1/pfor/TensorArrayV2TensorListReserveNloop_body/stateful_uniform/Bitcast_1/pfor/TensorArrayV2/element_shape:output:0@loop_body/stateful_uniform/Bitcast_1/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÌq
/loop_body/stateful_uniform/Bitcast_1/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
Bloop_body/stateful_uniform/Bitcast_1/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ~
<loop_body/stateful_uniform/Bitcast_1/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Û
/loop_body/stateful_uniform/Bitcast_1/pfor/whileStatelessWhileEloop_body/stateful_uniform/Bitcast_1/pfor/while/loop_counter:output:0Kloop_body/stateful_uniform/Bitcast_1/pfor/while/maximum_iterations:output:08loop_body/stateful_uniform/Bitcast_1/pfor/Const:output:0@loop_body/stateful_uniform/Bitcast_1/pfor/TensorArrayV2:handle:0@loop_body/stateful_uniform/Bitcast_1/pfor/strided_slice:output:0Eloop_body/stateful_uniform/strided_slice_1/pfor/StridedSlice:output:0*
T

2	*
_lower_using_switch_merge(*
_num_original_outputs*1
_output_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *
_stateful_parallelism( *E
body=R;
9loop_body_stateful_uniform_Bitcast_1_pfor_while_body_8167*E
cond=R;
9loop_body_stateful_uniform_Bitcast_1_pfor_while_cond_8166*0
output_shapes
: : : : : :ÿÿÿÿÿÿÿÿÿt
1loop_body/stateful_uniform/Bitcast_1/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 
Jloop_body/stateful_uniform/Bitcast_1/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ü
<loop_body/stateful_uniform/Bitcast_1/pfor/TensorListConcatV2TensorListConcatV28loop_body/stateful_uniform/Bitcast_1/pfor/while:output:3Sloop_body/stateful_uniform/Bitcast_1/pfor/TensorListConcatV2/element_shape:output:0:loop_body/stateful_uniform/Bitcast_1/pfor/Const_1:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0
Lloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Nloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Nloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¼
Floop_body/stateful_uniform/StatelessRandomUniformV2/pfor/strided_sliceStridedSlicepfor/Reshape:output:0Uloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/strided_slice/stack:output:0Wloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/strided_slice/stack_1:output:0Wloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Tloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÝ
Floop_body/stateful_uniform/StatelessRandomUniformV2/pfor/TensorArrayV2TensorListReserve]loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/TensorArrayV2/element_shape:output:0Oloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
>loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
Qloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Kloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¾	
>loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/whileStatelessWhileTloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/loop_counter:output:0Zloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while/maximum_iterations:output:0Gloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/Const:output:0Oloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/TensorArrayV2:handle:0Oloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/strided_slice:output:0Eloop_body/stateful_uniform/Bitcast_1/pfor/TensorListConcatV2:tensor:0Cloop_body/stateful_uniform/Bitcast/pfor/TensorListConcatV2:tensor:0)loop_body/stateful_uniform/shape:output:0@loop_body/stateful_uniform/StatelessRandomUniformV2/alg:output:0*
T
2	*
_lower_using_switch_merge(*
_num_original_outputs	*L
_output_shapes:
8: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: * 
_read_only_resource_inputs
 *
_stateful_parallelism( *T
bodyLRJ
Hloop_body_stateful_uniform_StatelessRandomUniformV2_pfor_while_body_8224*T
condLRJ
Hloop_body_stateful_uniform_StatelessRandomUniformV2_pfor_while_cond_8223*K
output_shapes:
8: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: 
@loop_body/stateful_uniform/StatelessRandomUniformV2/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 ª
Yloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿÿÿÿÿ¸
Kloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/TensorListConcatV2TensorListConcatV2Gloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/while:output:3bloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/TensorListConcatV2/element_shape:output:0Iloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/Const_1:output:0*6
_output_shapes$
":ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0j
(loop_body/stateful_uniform/mul/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :l
*loop_body/stateful_uniform/mul/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : k
)loop_body/stateful_uniform/mul/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :º
'loop_body/stateful_uniform/mul/pfor/addAddV23loop_body/stateful_uniform/mul/pfor/Rank_1:output:02loop_body/stateful_uniform/mul/pfor/add/y:output:0*
T0*
_output_shapes
: ·
+loop_body/stateful_uniform/mul/pfor/MaximumMaximum+loop_body/stateful_uniform/mul/pfor/add:z:01loop_body/stateful_uniform/mul/pfor/Rank:output:0*
T0*
_output_shapes
: ­
)loop_body/stateful_uniform/mul/pfor/ShapeShapeTloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/TensorListConcatV2:tensor:0*
T0*
_output_shapes
:³
'loop_body/stateful_uniform/mul/pfor/subSub/loop_body/stateful_uniform/mul/pfor/Maximum:z:01loop_body/stateful_uniform/mul/pfor/Rank:output:0*
T0*
_output_shapes
: {
1loop_body/stateful_uniform/mul/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:Ä
+loop_body/stateful_uniform/mul/pfor/ReshapeReshape+loop_body/stateful_uniform/mul/pfor/sub:z:0:loop_body/stateful_uniform/mul/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:x
.loop_body/stateful_uniform/mul/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:Â
(loop_body/stateful_uniform/mul/pfor/TileTile7loop_body/stateful_uniform/mul/pfor/Tile/input:output:04loop_body/stateful_uniform/mul/pfor/Reshape:output:0*
T0*
_output_shapes
: 
7loop_body/stateful_uniform/mul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
9loop_body/stateful_uniform/mul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
9loop_body/stateful_uniform/mul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
1loop_body/stateful_uniform/mul/pfor/strided_sliceStridedSlice2loop_body/stateful_uniform/mul/pfor/Shape:output:0@loop_body/stateful_uniform/mul/pfor/strided_slice/stack:output:0Bloop_body/stateful_uniform/mul/pfor/strided_slice/stack_1:output:0Bloop_body/stateful_uniform/mul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
9loop_body/stateful_uniform/mul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
;loop_body/stateful_uniform/mul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
;loop_body/stateful_uniform/mul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
3loop_body/stateful_uniform/mul/pfor/strided_slice_1StridedSlice2loop_body/stateful_uniform/mul/pfor/Shape:output:0Bloop_body/stateful_uniform/mul/pfor/strided_slice_1/stack:output:0Dloop_body/stateful_uniform/mul/pfor/strided_slice_1/stack_1:output:0Dloop_body/stateful_uniform/mul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskq
/loop_body/stateful_uniform/mul/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ë
*loop_body/stateful_uniform/mul/pfor/concatConcatV2:loop_body/stateful_uniform/mul/pfor/strided_slice:output:01loop_body/stateful_uniform/mul/pfor/Tile:output:0<loop_body/stateful_uniform/mul/pfor/strided_slice_1:output:08loop_body/stateful_uniform/mul/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:õ
-loop_body/stateful_uniform/mul/pfor/Reshape_1ReshapeTloop_body/stateful_uniform/StatelessRandomUniformV2/pfor/TensorListConcatV2:tensor:03loop_body/stateful_uniform/mul/pfor/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
'loop_body/stateful_uniform/mul/pfor/MulMul6loop_body/stateful_uniform/mul/pfor/Reshape_1:output:0"loop_body/stateful_uniform/sub:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
$loop_body/stateful_uniform/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :h
&loop_body/stateful_uniform/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : g
%loop_body/stateful_uniform/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :®
#loop_body/stateful_uniform/pfor/addAddV2/loop_body/stateful_uniform/pfor/Rank_1:output:0.loop_body/stateful_uniform/pfor/add/y:output:0*
T0*
_output_shapes
: «
'loop_body/stateful_uniform/pfor/MaximumMaximum'loop_body/stateful_uniform/pfor/add:z:0-loop_body/stateful_uniform/pfor/Rank:output:0*
T0*
_output_shapes
: 
%loop_body/stateful_uniform/pfor/ShapeShape+loop_body/stateful_uniform/mul/pfor/Mul:z:0*
T0*
_output_shapes
:§
#loop_body/stateful_uniform/pfor/subSub+loop_body/stateful_uniform/pfor/Maximum:z:0-loop_body/stateful_uniform/pfor/Rank:output:0*
T0*
_output_shapes
: w
-loop_body/stateful_uniform/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:¸
'loop_body/stateful_uniform/pfor/ReshapeReshape'loop_body/stateful_uniform/pfor/sub:z:06loop_body/stateful_uniform/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:t
*loop_body/stateful_uniform/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:¶
$loop_body/stateful_uniform/pfor/TileTile3loop_body/stateful_uniform/pfor/Tile/input:output:00loop_body/stateful_uniform/pfor/Reshape:output:0*
T0*
_output_shapes
: }
3loop_body/stateful_uniform/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5loop_body/stateful_uniform/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5loop_body/stateful_uniform/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
-loop_body/stateful_uniform/pfor/strided_sliceStridedSlice.loop_body/stateful_uniform/pfor/Shape:output:0<loop_body/stateful_uniform/pfor/strided_slice/stack:output:0>loop_body/stateful_uniform/pfor/strided_slice/stack_1:output:0>loop_body/stateful_uniform/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
5loop_body/stateful_uniform/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
7loop_body/stateful_uniform/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
7loop_body/stateful_uniform/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:õ
/loop_body/stateful_uniform/pfor/strided_slice_1StridedSlice.loop_body/stateful_uniform/pfor/Shape:output:0>loop_body/stateful_uniform/pfor/strided_slice_1/stack:output:0@loop_body/stateful_uniform/pfor/strided_slice_1/stack_1:output:0@loop_body/stateful_uniform/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskm
+loop_body/stateful_uniform/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ·
&loop_body/stateful_uniform/pfor/concatConcatV26loop_body/stateful_uniform/pfor/strided_slice:output:0-loop_body/stateful_uniform/pfor/Tile:output:08loop_body/stateful_uniform/pfor/strided_slice_1:output:04loop_body/stateful_uniform/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Ä
)loop_body/stateful_uniform/pfor/Reshape_1Reshape+loop_body/stateful_uniform/mul/pfor/Mul:z:0/loop_body/stateful_uniform/pfor/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
%loop_body/stateful_uniform/pfor/AddV2AddV22loop_body/stateful_uniform/pfor/Reshape_1:output:0'loop_body/stateful_uniform/min:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(loop_body/rotation_matrix/Cos_1/pfor/CosCos)loop_body/stateful_uniform/pfor/AddV2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
)loop_body/rotation_matrix/mul_3/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :m
+loop_body/rotation_matrix/mul_3/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : l
*loop_body/rotation_matrix/mul_3/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :½
(loop_body/rotation_matrix/mul_3/pfor/addAddV24loop_body/rotation_matrix/mul_3/pfor/Rank_1:output:03loop_body/rotation_matrix/mul_3/pfor/add/y:output:0*
T0*
_output_shapes
: º
,loop_body/rotation_matrix/mul_3/pfor/MaximumMaximum,loop_body/rotation_matrix/mul_3/pfor/add:z:02loop_body/rotation_matrix/mul_3/pfor/Rank:output:0*
T0*
_output_shapes
: 
*loop_body/rotation_matrix/mul_3/pfor/ShapeShape,loop_body/rotation_matrix/Cos_1/pfor/Cos:y:0*
T0*
_output_shapes
:¶
(loop_body/rotation_matrix/mul_3/pfor/subSub0loop_body/rotation_matrix/mul_3/pfor/Maximum:z:02loop_body/rotation_matrix/mul_3/pfor/Rank:output:0*
T0*
_output_shapes
: |
2loop_body/rotation_matrix/mul_3/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:Ç
,loop_body/rotation_matrix/mul_3/pfor/ReshapeReshape,loop_body/rotation_matrix/mul_3/pfor/sub:z:0;loop_body/rotation_matrix/mul_3/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:y
/loop_body/rotation_matrix/mul_3/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:Å
)loop_body/rotation_matrix/mul_3/pfor/TileTile8loop_body/rotation_matrix/mul_3/pfor/Tile/input:output:05loop_body/rotation_matrix/mul_3/pfor/Reshape:output:0*
T0*
_output_shapes
: 
8loop_body/rotation_matrix/mul_3/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
:loop_body/rotation_matrix/mul_3/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
:loop_body/rotation_matrix/mul_3/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
2loop_body/rotation_matrix/mul_3/pfor/strided_sliceStridedSlice3loop_body/rotation_matrix/mul_3/pfor/Shape:output:0Aloop_body/rotation_matrix/mul_3/pfor/strided_slice/stack:output:0Cloop_body/rotation_matrix/mul_3/pfor/strided_slice/stack_1:output:0Cloop_body/rotation_matrix/mul_3/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
:loop_body/rotation_matrix/mul_3/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
<loop_body/rotation_matrix/mul_3/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
<loop_body/rotation_matrix/mul_3/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
4loop_body/rotation_matrix/mul_3/pfor/strided_slice_1StridedSlice3loop_body/rotation_matrix/mul_3/pfor/Shape:output:0Cloop_body/rotation_matrix/mul_3/pfor/strided_slice_1/stack:output:0Eloop_body/rotation_matrix/mul_3/pfor/strided_slice_1/stack_1:output:0Eloop_body/rotation_matrix/mul_3/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskr
0loop_body/rotation_matrix/mul_3/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ð
+loop_body/rotation_matrix/mul_3/pfor/concatConcatV2;loop_body/rotation_matrix/mul_3/pfor/strided_slice:output:02loop_body/rotation_matrix/mul_3/pfor/Tile:output:0=loop_body/rotation_matrix/mul_3/pfor/strided_slice_1:output:09loop_body/rotation_matrix/mul_3/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Ï
.loop_body/rotation_matrix/mul_3/pfor/Reshape_1Reshape,loop_body/rotation_matrix/Cos_1/pfor/Cos:y:04loop_body/rotation_matrix/mul_3/pfor/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
(loop_body/rotation_matrix/mul_3/pfor/MulMul7loop_body/rotation_matrix/mul_3/pfor/Reshape_1:output:0#loop_body/rotation_matrix/sub_7:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(loop_body/rotation_matrix/Sin_1/pfor/SinSin)loop_body/stateful_uniform/pfor/AddV2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
)loop_body/rotation_matrix/mul_2/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :m
+loop_body/rotation_matrix/mul_2/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : l
*loop_body/rotation_matrix/mul_2/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :½
(loop_body/rotation_matrix/mul_2/pfor/addAddV24loop_body/rotation_matrix/mul_2/pfor/Rank_1:output:03loop_body/rotation_matrix/mul_2/pfor/add/y:output:0*
T0*
_output_shapes
: º
,loop_body/rotation_matrix/mul_2/pfor/MaximumMaximum,loop_body/rotation_matrix/mul_2/pfor/add:z:02loop_body/rotation_matrix/mul_2/pfor/Rank:output:0*
T0*
_output_shapes
: 
*loop_body/rotation_matrix/mul_2/pfor/ShapeShape,loop_body/rotation_matrix/Sin_1/pfor/Sin:y:0*
T0*
_output_shapes
:¶
(loop_body/rotation_matrix/mul_2/pfor/subSub0loop_body/rotation_matrix/mul_2/pfor/Maximum:z:02loop_body/rotation_matrix/mul_2/pfor/Rank:output:0*
T0*
_output_shapes
: |
2loop_body/rotation_matrix/mul_2/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:Ç
,loop_body/rotation_matrix/mul_2/pfor/ReshapeReshape,loop_body/rotation_matrix/mul_2/pfor/sub:z:0;loop_body/rotation_matrix/mul_2/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:y
/loop_body/rotation_matrix/mul_2/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:Å
)loop_body/rotation_matrix/mul_2/pfor/TileTile8loop_body/rotation_matrix/mul_2/pfor/Tile/input:output:05loop_body/rotation_matrix/mul_2/pfor/Reshape:output:0*
T0*
_output_shapes
: 
8loop_body/rotation_matrix/mul_2/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
:loop_body/rotation_matrix/mul_2/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
:loop_body/rotation_matrix/mul_2/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
2loop_body/rotation_matrix/mul_2/pfor/strided_sliceStridedSlice3loop_body/rotation_matrix/mul_2/pfor/Shape:output:0Aloop_body/rotation_matrix/mul_2/pfor/strided_slice/stack:output:0Cloop_body/rotation_matrix/mul_2/pfor/strided_slice/stack_1:output:0Cloop_body/rotation_matrix/mul_2/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
:loop_body/rotation_matrix/mul_2/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
<loop_body/rotation_matrix/mul_2/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
<loop_body/rotation_matrix/mul_2/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
4loop_body/rotation_matrix/mul_2/pfor/strided_slice_1StridedSlice3loop_body/rotation_matrix/mul_2/pfor/Shape:output:0Cloop_body/rotation_matrix/mul_2/pfor/strided_slice_1/stack:output:0Eloop_body/rotation_matrix/mul_2/pfor/strided_slice_1/stack_1:output:0Eloop_body/rotation_matrix/mul_2/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskr
0loop_body/rotation_matrix/mul_2/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ð
+loop_body/rotation_matrix/mul_2/pfor/concatConcatV2;loop_body/rotation_matrix/mul_2/pfor/strided_slice:output:02loop_body/rotation_matrix/mul_2/pfor/Tile:output:0=loop_body/rotation_matrix/mul_2/pfor/strided_slice_1:output:09loop_body/rotation_matrix/mul_2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Ï
.loop_body/rotation_matrix/mul_2/pfor/Reshape_1Reshape,loop_body/rotation_matrix/Sin_1/pfor/Sin:y:04loop_body/rotation_matrix/mul_2/pfor/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
(loop_body/rotation_matrix/mul_2/pfor/MulMul7loop_body/rotation_matrix/mul_2/pfor/Reshape_1:output:0#loop_body/rotation_matrix/sub_6:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
'loop_body/rotation_matrix/add/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :k
)loop_body/rotation_matrix/add/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :¼
*loop_body/rotation_matrix/add/pfor/MaximumMaximum2loop_body/rotation_matrix/add/pfor/Rank_1:output:00loop_body/rotation_matrix/add/pfor/Rank:output:0*
T0*
_output_shapes
: 
(loop_body/rotation_matrix/add/pfor/ShapeShape,loop_body/rotation_matrix/mul_2/pfor/Mul:z:0*
T0*
_output_shapes
:°
&loop_body/rotation_matrix/add/pfor/subSub.loop_body/rotation_matrix/add/pfor/Maximum:z:00loop_body/rotation_matrix/add/pfor/Rank:output:0*
T0*
_output_shapes
: z
0loop_body/rotation_matrix/add/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:Á
*loop_body/rotation_matrix/add/pfor/ReshapeReshape*loop_body/rotation_matrix/add/pfor/sub:z:09loop_body/rotation_matrix/add/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:w
-loop_body/rotation_matrix/add/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:¿
'loop_body/rotation_matrix/add/pfor/TileTile6loop_body/rotation_matrix/add/pfor/Tile/input:output:03loop_body/rotation_matrix/add/pfor/Reshape:output:0*
T0*
_output_shapes
: 
6loop_body/rotation_matrix/add/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
8loop_body/rotation_matrix/add/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
8loop_body/rotation_matrix/add/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:þ
0loop_body/rotation_matrix/add/pfor/strided_sliceStridedSlice1loop_body/rotation_matrix/add/pfor/Shape:output:0?loop_body/rotation_matrix/add/pfor/strided_slice/stack:output:0Aloop_body/rotation_matrix/add/pfor/strided_slice/stack_1:output:0Aloop_body/rotation_matrix/add/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
8loop_body/rotation_matrix/add/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
:loop_body/rotation_matrix/add/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
:loop_body/rotation_matrix/add/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
2loop_body/rotation_matrix/add/pfor/strided_slice_1StridedSlice1loop_body/rotation_matrix/add/pfor/Shape:output:0Aloop_body/rotation_matrix/add/pfor/strided_slice_1/stack:output:0Cloop_body/rotation_matrix/add/pfor/strided_slice_1/stack_1:output:0Cloop_body/rotation_matrix/add/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskp
.loop_body/rotation_matrix/add/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Æ
)loop_body/rotation_matrix/add/pfor/concatConcatV29loop_body/rotation_matrix/add/pfor/strided_slice:output:00loop_body/rotation_matrix/add/pfor/Tile:output:0;loop_body/rotation_matrix/add/pfor/strided_slice_1:output:07loop_body/rotation_matrix/add/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Ë
,loop_body/rotation_matrix/add/pfor/Reshape_1Reshape,loop_body/rotation_matrix/mul_2/pfor/Mul:z:02loop_body/rotation_matrix/add/pfor/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*loop_body/rotation_matrix/add/pfor/Shape_1Shape,loop_body/rotation_matrix/mul_3/pfor/Mul:z:0*
T0*
_output_shapes
:´
(loop_body/rotation_matrix/add/pfor/sub_1Sub.loop_body/rotation_matrix/add/pfor/Maximum:z:02loop_body/rotation_matrix/add/pfor/Rank_1:output:0*
T0*
_output_shapes
: |
2loop_body/rotation_matrix/add/pfor/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB:Ç
,loop_body/rotation_matrix/add/pfor/Reshape_2Reshape,loop_body/rotation_matrix/add/pfor/sub_1:z:0;loop_body/rotation_matrix/add/pfor/Reshape_2/shape:output:0*
T0*
_output_shapes
:y
/loop_body/rotation_matrix/add/pfor/Tile_1/inputConst*
_output_shapes
:*
dtype0*
valueB:Å
)loop_body/rotation_matrix/add/pfor/Tile_1Tile8loop_body/rotation_matrix/add/pfor/Tile_1/input:output:05loop_body/rotation_matrix/add/pfor/Reshape_2:output:0*
T0*
_output_shapes
: 
8loop_body/rotation_matrix/add/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
:loop_body/rotation_matrix/add/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
:loop_body/rotation_matrix/add/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
2loop_body/rotation_matrix/add/pfor/strided_slice_2StridedSlice3loop_body/rotation_matrix/add/pfor/Shape_1:output:0Aloop_body/rotation_matrix/add/pfor/strided_slice_2/stack:output:0Cloop_body/rotation_matrix/add/pfor/strided_slice_2/stack_1:output:0Cloop_body/rotation_matrix/add/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
8loop_body/rotation_matrix/add/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
:loop_body/rotation_matrix/add/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
:loop_body/rotation_matrix/add/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
2loop_body/rotation_matrix/add/pfor/strided_slice_3StridedSlice3loop_body/rotation_matrix/add/pfor/Shape_1:output:0Aloop_body/rotation_matrix/add/pfor/strided_slice_3/stack:output:0Cloop_body/rotation_matrix/add/pfor/strided_slice_3/stack_1:output:0Cloop_body/rotation_matrix/add/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskr
0loop_body/rotation_matrix/add/pfor/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Î
+loop_body/rotation_matrix/add/pfor/concat_1ConcatV2;loop_body/rotation_matrix/add/pfor/strided_slice_2:output:02loop_body/rotation_matrix/add/pfor/Tile_1:output:0;loop_body/rotation_matrix/add/pfor/strided_slice_3:output:09loop_body/rotation_matrix/add/pfor/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Í
,loop_body/rotation_matrix/add/pfor/Reshape_3Reshape,loop_body/rotation_matrix/mul_3/pfor/Mul:z:04loop_body/rotation_matrix/add/pfor/concat_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
(loop_body/rotation_matrix/add/pfor/AddV2AddV25loop_body/rotation_matrix/add/pfor/Reshape_1:output:05loop_body/rotation_matrix/add/pfor/Reshape_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
)loop_body/rotation_matrix/sub_8/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B : l
*loop_body/rotation_matrix/sub_8/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :»
(loop_body/rotation_matrix/sub_8/pfor/addAddV22loop_body/rotation_matrix/sub_8/pfor/Rank:output:03loop_body/rotation_matrix/sub_8/pfor/add/y:output:0*
T0*
_output_shapes
: m
+loop_body/rotation_matrix/sub_8/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :¼
,loop_body/rotation_matrix/sub_8/pfor/MaximumMaximum4loop_body/rotation_matrix/sub_8/pfor/Rank_1:output:0,loop_body/rotation_matrix/sub_8/pfor/add:z:0*
T0*
_output_shapes
: 
*loop_body/rotation_matrix/sub_8/pfor/ShapeShape,loop_body/rotation_matrix/add/pfor/AddV2:z:0*
T0*
_output_shapes
:¸
(loop_body/rotation_matrix/sub_8/pfor/subSub0loop_body/rotation_matrix/sub_8/pfor/Maximum:z:04loop_body/rotation_matrix/sub_8/pfor/Rank_1:output:0*
T0*
_output_shapes
: |
2loop_body/rotation_matrix/sub_8/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:Ç
,loop_body/rotation_matrix/sub_8/pfor/ReshapeReshape,loop_body/rotation_matrix/sub_8/pfor/sub:z:0;loop_body/rotation_matrix/sub_8/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:y
/loop_body/rotation_matrix/sub_8/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:Å
)loop_body/rotation_matrix/sub_8/pfor/TileTile8loop_body/rotation_matrix/sub_8/pfor/Tile/input:output:05loop_body/rotation_matrix/sub_8/pfor/Reshape:output:0*
T0*
_output_shapes
: 
8loop_body/rotation_matrix/sub_8/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
:loop_body/rotation_matrix/sub_8/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
:loop_body/rotation_matrix/sub_8/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
2loop_body/rotation_matrix/sub_8/pfor/strided_sliceStridedSlice3loop_body/rotation_matrix/sub_8/pfor/Shape:output:0Aloop_body/rotation_matrix/sub_8/pfor/strided_slice/stack:output:0Cloop_body/rotation_matrix/sub_8/pfor/strided_slice/stack_1:output:0Cloop_body/rotation_matrix/sub_8/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
:loop_body/rotation_matrix/sub_8/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
<loop_body/rotation_matrix/sub_8/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
<loop_body/rotation_matrix/sub_8/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
4loop_body/rotation_matrix/sub_8/pfor/strided_slice_1StridedSlice3loop_body/rotation_matrix/sub_8/pfor/Shape:output:0Cloop_body/rotation_matrix/sub_8/pfor/strided_slice_1/stack:output:0Eloop_body/rotation_matrix/sub_8/pfor/strided_slice_1/stack_1:output:0Eloop_body/rotation_matrix/sub_8/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskr
0loop_body/rotation_matrix/sub_8/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ð
+loop_body/rotation_matrix/sub_8/pfor/concatConcatV2;loop_body/rotation_matrix/sub_8/pfor/strided_slice:output:02loop_body/rotation_matrix/sub_8/pfor/Tile:output:0=loop_body/rotation_matrix/sub_8/pfor/strided_slice_1:output:09loop_body/rotation_matrix/sub_8/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Ï
.loop_body/rotation_matrix/sub_8/pfor/Reshape_1Reshape,loop_body/rotation_matrix/add/pfor/AddV2:z:04loop_body/rotation_matrix/sub_8/pfor/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÁ
*loop_body/rotation_matrix/sub_8/pfor/Sub_1Sub#loop_body/rotation_matrix/sub_5:z:07loop_body/rotation_matrix/sub_8/pfor/Reshape_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
-loop_body/rotation_matrix/truediv_1/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :q
/loop_body/rotation_matrix/truediv_1/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : p
.loop_body/rotation_matrix/truediv_1/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :É
,loop_body/rotation_matrix/truediv_1/pfor/addAddV28loop_body/rotation_matrix/truediv_1/pfor/Rank_1:output:07loop_body/rotation_matrix/truediv_1/pfor/add/y:output:0*
T0*
_output_shapes
: Æ
0loop_body/rotation_matrix/truediv_1/pfor/MaximumMaximum0loop_body/rotation_matrix/truediv_1/pfor/add:z:06loop_body/rotation_matrix/truediv_1/pfor/Rank:output:0*
T0*
_output_shapes
: 
.loop_body/rotation_matrix/truediv_1/pfor/ShapeShape.loop_body/rotation_matrix/sub_8/pfor/Sub_1:z:0*
T0*
_output_shapes
:Â
,loop_body/rotation_matrix/truediv_1/pfor/subSub4loop_body/rotation_matrix/truediv_1/pfor/Maximum:z:06loop_body/rotation_matrix/truediv_1/pfor/Rank:output:0*
T0*
_output_shapes
: 
6loop_body/rotation_matrix/truediv_1/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:Ó
0loop_body/rotation_matrix/truediv_1/pfor/ReshapeReshape0loop_body/rotation_matrix/truediv_1/pfor/sub:z:0?loop_body/rotation_matrix/truediv_1/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:}
3loop_body/rotation_matrix/truediv_1/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:Ñ
-loop_body/rotation_matrix/truediv_1/pfor/TileTile<loop_body/rotation_matrix/truediv_1/pfor/Tile/input:output:09loop_body/rotation_matrix/truediv_1/pfor/Reshape:output:0*
T0*
_output_shapes
: 
<loop_body/rotation_matrix/truediv_1/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
>loop_body/rotation_matrix/truediv_1/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
>loop_body/rotation_matrix/truediv_1/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
6loop_body/rotation_matrix/truediv_1/pfor/strided_sliceStridedSlice7loop_body/rotation_matrix/truediv_1/pfor/Shape:output:0Eloop_body/rotation_matrix/truediv_1/pfor/strided_slice/stack:output:0Gloop_body/rotation_matrix/truediv_1/pfor/strided_slice/stack_1:output:0Gloop_body/rotation_matrix/truediv_1/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
>loop_body/rotation_matrix/truediv_1/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
@loop_body/rotation_matrix/truediv_1/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
@loop_body/rotation_matrix/truediv_1/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¢
8loop_body/rotation_matrix/truediv_1/pfor/strided_slice_1StridedSlice7loop_body/rotation_matrix/truediv_1/pfor/Shape:output:0Gloop_body/rotation_matrix/truediv_1/pfor/strided_slice_1/stack:output:0Iloop_body/rotation_matrix/truediv_1/pfor/strided_slice_1/stack_1:output:0Iloop_body/rotation_matrix/truediv_1/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskv
4loop_body/rotation_matrix/truediv_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ä
/loop_body/rotation_matrix/truediv_1/pfor/concatConcatV2?loop_body/rotation_matrix/truediv_1/pfor/strided_slice:output:06loop_body/rotation_matrix/truediv_1/pfor/Tile:output:0Aloop_body/rotation_matrix/truediv_1/pfor/strided_slice_1:output:0=loop_body/rotation_matrix/truediv_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Ù
2loop_body/rotation_matrix/truediv_1/pfor/Reshape_1Reshape.loop_body/rotation_matrix/sub_8/pfor/Sub_1:z:08loop_body/rotation_matrix/truediv_1/pfor/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
0loop_body/rotation_matrix/truediv_1/pfor/RealDivRealDiv;loop_body/rotation_matrix/truediv_1/pfor/Reshape_1:output:0.loop_body/rotation_matrix/truediv_1/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
>loop_body/rotation_matrix/strided_slice_6/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: |
:loop_body/rotation_matrix/strided_slice_6/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ·
5loop_body/rotation_matrix/strided_slice_6/pfor/concatConcatV2Gloop_body/rotation_matrix/strided_slice_6/pfor/concat/values_0:output:08loop_body/rotation_matrix/strided_slice_6/stack:output:0Cloop_body/rotation_matrix/strided_slice_6/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
@loop_body/rotation_matrix/strided_slice_6/pfor/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB: ~
<loop_body/rotation_matrix/strided_slice_6/pfor/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
7loop_body/rotation_matrix/strided_slice_6/pfor/concat_1ConcatV2Iloop_body/rotation_matrix/strided_slice_6/pfor/concat_1/values_0:output:0:loop_body/rotation_matrix/strided_slice_6/stack_1:output:0Eloop_body/rotation_matrix/strided_slice_6/pfor/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
@loop_body/rotation_matrix/strided_slice_6/pfor/concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB:~
<loop_body/rotation_matrix/strided_slice_6/pfor/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
7loop_body/rotation_matrix/strided_slice_6/pfor/concat_2ConcatV2Iloop_body/rotation_matrix/strided_slice_6/pfor/concat_2/values_0:output:0:loop_body/rotation_matrix/strided_slice_6/stack_2:output:0Eloop_body/rotation_matrix/strided_slice_6/pfor/concat_2/axis:output:0*
N*
T0*
_output_shapes
:¿
;loop_body/rotation_matrix/strided_slice_6/pfor/StridedSliceStridedSlice4loop_body/rotation_matrix/truediv_1/pfor/RealDiv:z:0>loop_body/rotation_matrix/strided_slice_6/pfor/concat:output:0@loop_body/rotation_matrix/strided_slice_6/pfor/concat_1:output:0@loop_body/rotation_matrix/strided_slice_6/pfor/concat_2:output:0*
Index0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask
(loop_body/rotation_matrix/Cos_3/pfor/CosCos)loop_body/stateful_uniform/pfor/AddV2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
>loop_body/rotation_matrix/strided_slice_5/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: |
:loop_body/rotation_matrix/strided_slice_5/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ·
5loop_body/rotation_matrix/strided_slice_5/pfor/concatConcatV2Gloop_body/rotation_matrix/strided_slice_5/pfor/concat/values_0:output:08loop_body/rotation_matrix/strided_slice_5/stack:output:0Cloop_body/rotation_matrix/strided_slice_5/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
@loop_body/rotation_matrix/strided_slice_5/pfor/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB: ~
<loop_body/rotation_matrix/strided_slice_5/pfor/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
7loop_body/rotation_matrix/strided_slice_5/pfor/concat_1ConcatV2Iloop_body/rotation_matrix/strided_slice_5/pfor/concat_1/values_0:output:0:loop_body/rotation_matrix/strided_slice_5/stack_1:output:0Eloop_body/rotation_matrix/strided_slice_5/pfor/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
@loop_body/rotation_matrix/strided_slice_5/pfor/concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB:~
<loop_body/rotation_matrix/strided_slice_5/pfor/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
7loop_body/rotation_matrix/strided_slice_5/pfor/concat_2ConcatV2Iloop_body/rotation_matrix/strided_slice_5/pfor/concat_2/values_0:output:0:loop_body/rotation_matrix/strided_slice_5/stack_2:output:0Eloop_body/rotation_matrix/strided_slice_5/pfor/concat_2/axis:output:0*
N*
T0*
_output_shapes
:·
;loop_body/rotation_matrix/strided_slice_5/pfor/StridedSliceStridedSlice,loop_body/rotation_matrix/Cos_3/pfor/Cos:y:0>loop_body/rotation_matrix/strided_slice_5/pfor/concat:output:0@loop_body/rotation_matrix/strided_slice_5/pfor/concat_1:output:0@loop_body/rotation_matrix/strided_slice_5/pfor/concat_2:output:0*
Index0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask
(loop_body/rotation_matrix/Sin_3/pfor/SinSin)loop_body/stateful_uniform/pfor/AddV2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
>loop_body/rotation_matrix/strided_slice_4/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: |
:loop_body/rotation_matrix/strided_slice_4/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ·
5loop_body/rotation_matrix/strided_slice_4/pfor/concatConcatV2Gloop_body/rotation_matrix/strided_slice_4/pfor/concat/values_0:output:08loop_body/rotation_matrix/strided_slice_4/stack:output:0Cloop_body/rotation_matrix/strided_slice_4/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
@loop_body/rotation_matrix/strided_slice_4/pfor/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB: ~
<loop_body/rotation_matrix/strided_slice_4/pfor/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
7loop_body/rotation_matrix/strided_slice_4/pfor/concat_1ConcatV2Iloop_body/rotation_matrix/strided_slice_4/pfor/concat_1/values_0:output:0:loop_body/rotation_matrix/strided_slice_4/stack_1:output:0Eloop_body/rotation_matrix/strided_slice_4/pfor/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
@loop_body/rotation_matrix/strided_slice_4/pfor/concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB:~
<loop_body/rotation_matrix/strided_slice_4/pfor/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
7loop_body/rotation_matrix/strided_slice_4/pfor/concat_2ConcatV2Iloop_body/rotation_matrix/strided_slice_4/pfor/concat_2/values_0:output:0:loop_body/rotation_matrix/strided_slice_4/stack_2:output:0Eloop_body/rotation_matrix/strided_slice_4/pfor/concat_2/axis:output:0*
N*
T0*
_output_shapes
:·
;loop_body/rotation_matrix/strided_slice_4/pfor/StridedSliceStridedSlice,loop_body/rotation_matrix/Sin_3/pfor/Sin:y:0>loop_body/rotation_matrix/strided_slice_4/pfor/concat:output:0@loop_body/rotation_matrix/strided_slice_4/pfor/concat_1:output:0@loop_body/rotation_matrix/strided_slice_4/pfor/concat_2:output:0*
Index0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask
&loop_body/rotation_matrix/Sin/pfor/SinSin)loop_body/stateful_uniform/pfor/AddV2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
)loop_body/rotation_matrix/mul_1/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :m
+loop_body/rotation_matrix/mul_1/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : l
*loop_body/rotation_matrix/mul_1/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :½
(loop_body/rotation_matrix/mul_1/pfor/addAddV24loop_body/rotation_matrix/mul_1/pfor/Rank_1:output:03loop_body/rotation_matrix/mul_1/pfor/add/y:output:0*
T0*
_output_shapes
: º
,loop_body/rotation_matrix/mul_1/pfor/MaximumMaximum,loop_body/rotation_matrix/mul_1/pfor/add:z:02loop_body/rotation_matrix/mul_1/pfor/Rank:output:0*
T0*
_output_shapes
: 
*loop_body/rotation_matrix/mul_1/pfor/ShapeShape*loop_body/rotation_matrix/Sin/pfor/Sin:y:0*
T0*
_output_shapes
:¶
(loop_body/rotation_matrix/mul_1/pfor/subSub0loop_body/rotation_matrix/mul_1/pfor/Maximum:z:02loop_body/rotation_matrix/mul_1/pfor/Rank:output:0*
T0*
_output_shapes
: |
2loop_body/rotation_matrix/mul_1/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:Ç
,loop_body/rotation_matrix/mul_1/pfor/ReshapeReshape,loop_body/rotation_matrix/mul_1/pfor/sub:z:0;loop_body/rotation_matrix/mul_1/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:y
/loop_body/rotation_matrix/mul_1/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:Å
)loop_body/rotation_matrix/mul_1/pfor/TileTile8loop_body/rotation_matrix/mul_1/pfor/Tile/input:output:05loop_body/rotation_matrix/mul_1/pfor/Reshape:output:0*
T0*
_output_shapes
: 
8loop_body/rotation_matrix/mul_1/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
:loop_body/rotation_matrix/mul_1/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
:loop_body/rotation_matrix/mul_1/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
2loop_body/rotation_matrix/mul_1/pfor/strided_sliceStridedSlice3loop_body/rotation_matrix/mul_1/pfor/Shape:output:0Aloop_body/rotation_matrix/mul_1/pfor/strided_slice/stack:output:0Cloop_body/rotation_matrix/mul_1/pfor/strided_slice/stack_1:output:0Cloop_body/rotation_matrix/mul_1/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
:loop_body/rotation_matrix/mul_1/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
<loop_body/rotation_matrix/mul_1/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
<loop_body/rotation_matrix/mul_1/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
4loop_body/rotation_matrix/mul_1/pfor/strided_slice_1StridedSlice3loop_body/rotation_matrix/mul_1/pfor/Shape:output:0Cloop_body/rotation_matrix/mul_1/pfor/strided_slice_1/stack:output:0Eloop_body/rotation_matrix/mul_1/pfor/strided_slice_1/stack_1:output:0Eloop_body/rotation_matrix/mul_1/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskr
0loop_body/rotation_matrix/mul_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ð
+loop_body/rotation_matrix/mul_1/pfor/concatConcatV2;loop_body/rotation_matrix/mul_1/pfor/strided_slice:output:02loop_body/rotation_matrix/mul_1/pfor/Tile:output:0=loop_body/rotation_matrix/mul_1/pfor/strided_slice_1:output:09loop_body/rotation_matrix/mul_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Í
.loop_body/rotation_matrix/mul_1/pfor/Reshape_1Reshape*loop_body/rotation_matrix/Sin/pfor/Sin:y:04loop_body/rotation_matrix/mul_1/pfor/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
(loop_body/rotation_matrix/mul_1/pfor/MulMul7loop_body/rotation_matrix/mul_1/pfor/Reshape_1:output:0#loop_body/rotation_matrix/sub_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&loop_body/rotation_matrix/Cos/pfor/CosCos)loop_body/stateful_uniform/pfor/AddV2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
'loop_body/rotation_matrix/mul/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :k
)loop_body/rotation_matrix/mul/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : j
(loop_body/rotation_matrix/mul/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :·
&loop_body/rotation_matrix/mul/pfor/addAddV22loop_body/rotation_matrix/mul/pfor/Rank_1:output:01loop_body/rotation_matrix/mul/pfor/add/y:output:0*
T0*
_output_shapes
: ´
*loop_body/rotation_matrix/mul/pfor/MaximumMaximum*loop_body/rotation_matrix/mul/pfor/add:z:00loop_body/rotation_matrix/mul/pfor/Rank:output:0*
T0*
_output_shapes
: 
(loop_body/rotation_matrix/mul/pfor/ShapeShape*loop_body/rotation_matrix/Cos/pfor/Cos:y:0*
T0*
_output_shapes
:°
&loop_body/rotation_matrix/mul/pfor/subSub.loop_body/rotation_matrix/mul/pfor/Maximum:z:00loop_body/rotation_matrix/mul/pfor/Rank:output:0*
T0*
_output_shapes
: z
0loop_body/rotation_matrix/mul/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:Á
*loop_body/rotation_matrix/mul/pfor/ReshapeReshape*loop_body/rotation_matrix/mul/pfor/sub:z:09loop_body/rotation_matrix/mul/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:w
-loop_body/rotation_matrix/mul/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:¿
'loop_body/rotation_matrix/mul/pfor/TileTile6loop_body/rotation_matrix/mul/pfor/Tile/input:output:03loop_body/rotation_matrix/mul/pfor/Reshape:output:0*
T0*
_output_shapes
: 
6loop_body/rotation_matrix/mul/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
8loop_body/rotation_matrix/mul/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
8loop_body/rotation_matrix/mul/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:þ
0loop_body/rotation_matrix/mul/pfor/strided_sliceStridedSlice1loop_body/rotation_matrix/mul/pfor/Shape:output:0?loop_body/rotation_matrix/mul/pfor/strided_slice/stack:output:0Aloop_body/rotation_matrix/mul/pfor/strided_slice/stack_1:output:0Aloop_body/rotation_matrix/mul/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
8loop_body/rotation_matrix/mul/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
:loop_body/rotation_matrix/mul/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
:loop_body/rotation_matrix/mul/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
2loop_body/rotation_matrix/mul/pfor/strided_slice_1StridedSlice1loop_body/rotation_matrix/mul/pfor/Shape:output:0Aloop_body/rotation_matrix/mul/pfor/strided_slice_1/stack:output:0Cloop_body/rotation_matrix/mul/pfor/strided_slice_1/stack_1:output:0Cloop_body/rotation_matrix/mul/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskp
.loop_body/rotation_matrix/mul/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Æ
)loop_body/rotation_matrix/mul/pfor/concatConcatV29loop_body/rotation_matrix/mul/pfor/strided_slice:output:00loop_body/rotation_matrix/mul/pfor/Tile:output:0;loop_body/rotation_matrix/mul/pfor/strided_slice_1:output:07loop_body/rotation_matrix/mul/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:É
,loop_body/rotation_matrix/mul/pfor/Reshape_1Reshape*loop_body/rotation_matrix/Cos/pfor/Cos:y:02loop_body/rotation_matrix/mul/pfor/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ»
&loop_body/rotation_matrix/mul/pfor/MulMul5loop_body/rotation_matrix/mul/pfor/Reshape_1:output:0#loop_body/rotation_matrix/sub_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
)loop_body/rotation_matrix/sub_3/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :m
+loop_body/rotation_matrix/sub_3/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :Â
,loop_body/rotation_matrix/sub_3/pfor/MaximumMaximum4loop_body/rotation_matrix/sub_3/pfor/Rank_1:output:02loop_body/rotation_matrix/sub_3/pfor/Rank:output:0*
T0*
_output_shapes
: 
*loop_body/rotation_matrix/sub_3/pfor/ShapeShape*loop_body/rotation_matrix/mul/pfor/Mul:z:0*
T0*
_output_shapes
:¶
(loop_body/rotation_matrix/sub_3/pfor/subSub0loop_body/rotation_matrix/sub_3/pfor/Maximum:z:02loop_body/rotation_matrix/sub_3/pfor/Rank:output:0*
T0*
_output_shapes
: |
2loop_body/rotation_matrix/sub_3/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:Ç
,loop_body/rotation_matrix/sub_3/pfor/ReshapeReshape,loop_body/rotation_matrix/sub_3/pfor/sub:z:0;loop_body/rotation_matrix/sub_3/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:y
/loop_body/rotation_matrix/sub_3/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:Å
)loop_body/rotation_matrix/sub_3/pfor/TileTile8loop_body/rotation_matrix/sub_3/pfor/Tile/input:output:05loop_body/rotation_matrix/sub_3/pfor/Reshape:output:0*
T0*
_output_shapes
: 
8loop_body/rotation_matrix/sub_3/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
:loop_body/rotation_matrix/sub_3/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
:loop_body/rotation_matrix/sub_3/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
2loop_body/rotation_matrix/sub_3/pfor/strided_sliceStridedSlice3loop_body/rotation_matrix/sub_3/pfor/Shape:output:0Aloop_body/rotation_matrix/sub_3/pfor/strided_slice/stack:output:0Cloop_body/rotation_matrix/sub_3/pfor/strided_slice/stack_1:output:0Cloop_body/rotation_matrix/sub_3/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
:loop_body/rotation_matrix/sub_3/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
<loop_body/rotation_matrix/sub_3/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
<loop_body/rotation_matrix/sub_3/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
4loop_body/rotation_matrix/sub_3/pfor/strided_slice_1StridedSlice3loop_body/rotation_matrix/sub_3/pfor/Shape:output:0Cloop_body/rotation_matrix/sub_3/pfor/strided_slice_1/stack:output:0Eloop_body/rotation_matrix/sub_3/pfor/strided_slice_1/stack_1:output:0Eloop_body/rotation_matrix/sub_3/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskr
0loop_body/rotation_matrix/sub_3/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ð
+loop_body/rotation_matrix/sub_3/pfor/concatConcatV2;loop_body/rotation_matrix/sub_3/pfor/strided_slice:output:02loop_body/rotation_matrix/sub_3/pfor/Tile:output:0=loop_body/rotation_matrix/sub_3/pfor/strided_slice_1:output:09loop_body/rotation_matrix/sub_3/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Í
.loop_body/rotation_matrix/sub_3/pfor/Reshape_1Reshape*loop_body/rotation_matrix/mul/pfor/Mul:z:04loop_body/rotation_matrix/sub_3/pfor/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,loop_body/rotation_matrix/sub_3/pfor/Shape_1Shape,loop_body/rotation_matrix/mul_1/pfor/Mul:z:0*
T0*
_output_shapes
:º
*loop_body/rotation_matrix/sub_3/pfor/sub_1Sub0loop_body/rotation_matrix/sub_3/pfor/Maximum:z:04loop_body/rotation_matrix/sub_3/pfor/Rank_1:output:0*
T0*
_output_shapes
: ~
4loop_body/rotation_matrix/sub_3/pfor/Reshape_2/shapeConst*
_output_shapes
:*
dtype0*
valueB:Í
.loop_body/rotation_matrix/sub_3/pfor/Reshape_2Reshape.loop_body/rotation_matrix/sub_3/pfor/sub_1:z:0=loop_body/rotation_matrix/sub_3/pfor/Reshape_2/shape:output:0*
T0*
_output_shapes
:{
1loop_body/rotation_matrix/sub_3/pfor/Tile_1/inputConst*
_output_shapes
:*
dtype0*
valueB:Ë
+loop_body/rotation_matrix/sub_3/pfor/Tile_1Tile:loop_body/rotation_matrix/sub_3/pfor/Tile_1/input:output:07loop_body/rotation_matrix/sub_3/pfor/Reshape_2:output:0*
T0*
_output_shapes
: 
:loop_body/rotation_matrix/sub_3/pfor/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: 
<loop_body/rotation_matrix/sub_3/pfor/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
<loop_body/rotation_matrix/sub_3/pfor/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
4loop_body/rotation_matrix/sub_3/pfor/strided_slice_2StridedSlice5loop_body/rotation_matrix/sub_3/pfor/Shape_1:output:0Cloop_body/rotation_matrix/sub_3/pfor/strided_slice_2/stack:output:0Eloop_body/rotation_matrix/sub_3/pfor/strided_slice_2/stack_1:output:0Eloop_body/rotation_matrix/sub_3/pfor/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
:loop_body/rotation_matrix/sub_3/pfor/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
<loop_body/rotation_matrix/sub_3/pfor/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
<loop_body/rotation_matrix/sub_3/pfor/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
4loop_body/rotation_matrix/sub_3/pfor/strided_slice_3StridedSlice5loop_body/rotation_matrix/sub_3/pfor/Shape_1:output:0Cloop_body/rotation_matrix/sub_3/pfor/strided_slice_3/stack:output:0Eloop_body/rotation_matrix/sub_3/pfor/strided_slice_3/stack_1:output:0Eloop_body/rotation_matrix/sub_3/pfor/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskt
2loop_body/rotation_matrix/sub_3/pfor/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ø
-loop_body/rotation_matrix/sub_3/pfor/concat_1ConcatV2=loop_body/rotation_matrix/sub_3/pfor/strided_slice_2:output:04loop_body/rotation_matrix/sub_3/pfor/Tile_1:output:0=loop_body/rotation_matrix/sub_3/pfor/strided_slice_3:output:0;loop_body/rotation_matrix/sub_3/pfor/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ñ
.loop_body/rotation_matrix/sub_3/pfor/Reshape_3Reshape,loop_body/rotation_matrix/mul_1/pfor/Mul:z:06loop_body/rotation_matrix/sub_3/pfor/concat_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ
*loop_body/rotation_matrix/sub_3/pfor/Sub_2Sub7loop_body/rotation_matrix/sub_3/pfor/Reshape_1:output:07loop_body/rotation_matrix/sub_3/pfor/Reshape_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
)loop_body/rotation_matrix/sub_4/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B : l
*loop_body/rotation_matrix/sub_4/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :»
(loop_body/rotation_matrix/sub_4/pfor/addAddV22loop_body/rotation_matrix/sub_4/pfor/Rank:output:03loop_body/rotation_matrix/sub_4/pfor/add/y:output:0*
T0*
_output_shapes
: m
+loop_body/rotation_matrix/sub_4/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :¼
,loop_body/rotation_matrix/sub_4/pfor/MaximumMaximum4loop_body/rotation_matrix/sub_4/pfor/Rank_1:output:0,loop_body/rotation_matrix/sub_4/pfor/add:z:0*
T0*
_output_shapes
: 
*loop_body/rotation_matrix/sub_4/pfor/ShapeShape.loop_body/rotation_matrix/sub_3/pfor/Sub_2:z:0*
T0*
_output_shapes
:¸
(loop_body/rotation_matrix/sub_4/pfor/subSub0loop_body/rotation_matrix/sub_4/pfor/Maximum:z:04loop_body/rotation_matrix/sub_4/pfor/Rank_1:output:0*
T0*
_output_shapes
: |
2loop_body/rotation_matrix/sub_4/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:Ç
,loop_body/rotation_matrix/sub_4/pfor/ReshapeReshape,loop_body/rotation_matrix/sub_4/pfor/sub:z:0;loop_body/rotation_matrix/sub_4/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:y
/loop_body/rotation_matrix/sub_4/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:Å
)loop_body/rotation_matrix/sub_4/pfor/TileTile8loop_body/rotation_matrix/sub_4/pfor/Tile/input:output:05loop_body/rotation_matrix/sub_4/pfor/Reshape:output:0*
T0*
_output_shapes
: 
8loop_body/rotation_matrix/sub_4/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
:loop_body/rotation_matrix/sub_4/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
:loop_body/rotation_matrix/sub_4/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
2loop_body/rotation_matrix/sub_4/pfor/strided_sliceStridedSlice3loop_body/rotation_matrix/sub_4/pfor/Shape:output:0Aloop_body/rotation_matrix/sub_4/pfor/strided_slice/stack:output:0Cloop_body/rotation_matrix/sub_4/pfor/strided_slice/stack_1:output:0Cloop_body/rotation_matrix/sub_4/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
:loop_body/rotation_matrix/sub_4/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
<loop_body/rotation_matrix/sub_4/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
<loop_body/rotation_matrix/sub_4/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
4loop_body/rotation_matrix/sub_4/pfor/strided_slice_1StridedSlice3loop_body/rotation_matrix/sub_4/pfor/Shape:output:0Cloop_body/rotation_matrix/sub_4/pfor/strided_slice_1/stack:output:0Eloop_body/rotation_matrix/sub_4/pfor/strided_slice_1/stack_1:output:0Eloop_body/rotation_matrix/sub_4/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskr
0loop_body/rotation_matrix/sub_4/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ð
+loop_body/rotation_matrix/sub_4/pfor/concatConcatV2;loop_body/rotation_matrix/sub_4/pfor/strided_slice:output:02loop_body/rotation_matrix/sub_4/pfor/Tile:output:0=loop_body/rotation_matrix/sub_4/pfor/strided_slice_1:output:09loop_body/rotation_matrix/sub_4/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Ñ
.loop_body/rotation_matrix/sub_4/pfor/Reshape_1Reshape.loop_body/rotation_matrix/sub_3/pfor/Sub_2:z:04loop_body/rotation_matrix/sub_4/pfor/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
*loop_body/rotation_matrix/sub_4/pfor/Sub_1Sub!loop_body/rotation_matrix/sub:z:07loop_body/rotation_matrix/sub_4/pfor/Reshape_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
+loop_body/rotation_matrix/truediv/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B :o
-loop_body/rotation_matrix/truediv/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B : n
,loop_body/rotation_matrix/truediv/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :Ã
*loop_body/rotation_matrix/truediv/pfor/addAddV26loop_body/rotation_matrix/truediv/pfor/Rank_1:output:05loop_body/rotation_matrix/truediv/pfor/add/y:output:0*
T0*
_output_shapes
: À
.loop_body/rotation_matrix/truediv/pfor/MaximumMaximum.loop_body/rotation_matrix/truediv/pfor/add:z:04loop_body/rotation_matrix/truediv/pfor/Rank:output:0*
T0*
_output_shapes
: 
,loop_body/rotation_matrix/truediv/pfor/ShapeShape.loop_body/rotation_matrix/sub_4/pfor/Sub_1:z:0*
T0*
_output_shapes
:¼
*loop_body/rotation_matrix/truediv/pfor/subSub2loop_body/rotation_matrix/truediv/pfor/Maximum:z:04loop_body/rotation_matrix/truediv/pfor/Rank:output:0*
T0*
_output_shapes
: ~
4loop_body/rotation_matrix/truediv/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:Í
.loop_body/rotation_matrix/truediv/pfor/ReshapeReshape.loop_body/rotation_matrix/truediv/pfor/sub:z:0=loop_body/rotation_matrix/truediv/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:{
1loop_body/rotation_matrix/truediv/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:Ë
+loop_body/rotation_matrix/truediv/pfor/TileTile:loop_body/rotation_matrix/truediv/pfor/Tile/input:output:07loop_body/rotation_matrix/truediv/pfor/Reshape:output:0*
T0*
_output_shapes
: 
:loop_body/rotation_matrix/truediv/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
<loop_body/rotation_matrix/truediv/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
<loop_body/rotation_matrix/truediv/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
4loop_body/rotation_matrix/truediv/pfor/strided_sliceStridedSlice5loop_body/rotation_matrix/truediv/pfor/Shape:output:0Cloop_body/rotation_matrix/truediv/pfor/strided_slice/stack:output:0Eloop_body/rotation_matrix/truediv/pfor/strided_slice/stack_1:output:0Eloop_body/rotation_matrix/truediv/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_mask
<loop_body/rotation_matrix/truediv/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:
>loop_body/rotation_matrix/truediv/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 
>loop_body/rotation_matrix/truediv/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
6loop_body/rotation_matrix/truediv/pfor/strided_slice_1StridedSlice5loop_body/rotation_matrix/truediv/pfor/Shape:output:0Eloop_body/rotation_matrix/truediv/pfor/strided_slice_1/stack:output:0Gloop_body/rotation_matrix/truediv/pfor/strided_slice_1/stack_1:output:0Gloop_body/rotation_matrix/truediv/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_maskt
2loop_body/rotation_matrix/truediv/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ú
-loop_body/rotation_matrix/truediv/pfor/concatConcatV2=loop_body/rotation_matrix/truediv/pfor/strided_slice:output:04loop_body/rotation_matrix/truediv/pfor/Tile:output:0?loop_body/rotation_matrix/truediv/pfor/strided_slice_1:output:0;loop_body/rotation_matrix/truediv/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:Õ
0loop_body/rotation_matrix/truediv/pfor/Reshape_1Reshape.loop_body/rotation_matrix/sub_4/pfor/Sub_1:z:06loop_body/rotation_matrix/truediv/pfor/concat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÔ
.loop_body/rotation_matrix/truediv/pfor/RealDivRealDiv9loop_body/rotation_matrix/truediv/pfor/Reshape_1:output:0,loop_body/rotation_matrix/truediv/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
>loop_body/rotation_matrix/strided_slice_3/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: |
:loop_body/rotation_matrix/strided_slice_3/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ·
5loop_body/rotation_matrix/strided_slice_3/pfor/concatConcatV2Gloop_body/rotation_matrix/strided_slice_3/pfor/concat/values_0:output:08loop_body/rotation_matrix/strided_slice_3/stack:output:0Cloop_body/rotation_matrix/strided_slice_3/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
@loop_body/rotation_matrix/strided_slice_3/pfor/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB: ~
<loop_body/rotation_matrix/strided_slice_3/pfor/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
7loop_body/rotation_matrix/strided_slice_3/pfor/concat_1ConcatV2Iloop_body/rotation_matrix/strided_slice_3/pfor/concat_1/values_0:output:0:loop_body/rotation_matrix/strided_slice_3/stack_1:output:0Eloop_body/rotation_matrix/strided_slice_3/pfor/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
@loop_body/rotation_matrix/strided_slice_3/pfor/concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB:~
<loop_body/rotation_matrix/strided_slice_3/pfor/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
7loop_body/rotation_matrix/strided_slice_3/pfor/concat_2ConcatV2Iloop_body/rotation_matrix/strided_slice_3/pfor/concat_2/values_0:output:0:loop_body/rotation_matrix/strided_slice_3/stack_2:output:0Eloop_body/rotation_matrix/strided_slice_3/pfor/concat_2/axis:output:0*
N*
T0*
_output_shapes
:½
;loop_body/rotation_matrix/strided_slice_3/pfor/StridedSliceStridedSlice2loop_body/rotation_matrix/truediv/pfor/RealDiv:z:0>loop_body/rotation_matrix/strided_slice_3/pfor/concat:output:0@loop_body/rotation_matrix/strided_slice_3/pfor/concat_1:output:0@loop_body/rotation_matrix/strided_slice_3/pfor/concat_2:output:0*
Index0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask
(loop_body/rotation_matrix/Sin_2/pfor/SinSin)loop_body/stateful_uniform/pfor/AddV2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
>loop_body/rotation_matrix/strided_slice_2/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: |
:loop_body/rotation_matrix/strided_slice_2/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ·
5loop_body/rotation_matrix/strided_slice_2/pfor/concatConcatV2Gloop_body/rotation_matrix/strided_slice_2/pfor/concat/values_0:output:08loop_body/rotation_matrix/strided_slice_2/stack:output:0Cloop_body/rotation_matrix/strided_slice_2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
@loop_body/rotation_matrix/strided_slice_2/pfor/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB: ~
<loop_body/rotation_matrix/strided_slice_2/pfor/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
7loop_body/rotation_matrix/strided_slice_2/pfor/concat_1ConcatV2Iloop_body/rotation_matrix/strided_slice_2/pfor/concat_1/values_0:output:0:loop_body/rotation_matrix/strided_slice_2/stack_1:output:0Eloop_body/rotation_matrix/strided_slice_2/pfor/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
@loop_body/rotation_matrix/strided_slice_2/pfor/concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB:~
<loop_body/rotation_matrix/strided_slice_2/pfor/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
7loop_body/rotation_matrix/strided_slice_2/pfor/concat_2ConcatV2Iloop_body/rotation_matrix/strided_slice_2/pfor/concat_2/values_0:output:0:loop_body/rotation_matrix/strided_slice_2/stack_2:output:0Eloop_body/rotation_matrix/strided_slice_2/pfor/concat_2/axis:output:0*
N*
T0*
_output_shapes
:·
;loop_body/rotation_matrix/strided_slice_2/pfor/StridedSliceStridedSlice,loop_body/rotation_matrix/Sin_2/pfor/Sin:y:0>loop_body/rotation_matrix/strided_slice_2/pfor/concat:output:0@loop_body/rotation_matrix/strided_slice_2/pfor/concat_1:output:0@loop_body/rotation_matrix/strided_slice_2/pfor/concat_2:output:0*
Index0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask©
&loop_body/rotation_matrix/Neg/pfor/NegNegDloop_body/rotation_matrix/strided_slice_2/pfor/StridedSlice:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(loop_body/rotation_matrix/Cos_2/pfor/CosCos)loop_body/stateful_uniform/pfor/AddV2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
>loop_body/rotation_matrix/strided_slice_1/pfor/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB: |
:loop_body/rotation_matrix/strided_slice_1/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ·
5loop_body/rotation_matrix/strided_slice_1/pfor/concatConcatV2Gloop_body/rotation_matrix/strided_slice_1/pfor/concat/values_0:output:08loop_body/rotation_matrix/strided_slice_1/stack:output:0Cloop_body/rotation_matrix/strided_slice_1/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
@loop_body/rotation_matrix/strided_slice_1/pfor/concat_1/values_0Const*
_output_shapes
:*
dtype0*
valueB: ~
<loop_body/rotation_matrix/strided_slice_1/pfor/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
7loop_body/rotation_matrix/strided_slice_1/pfor/concat_1ConcatV2Iloop_body/rotation_matrix/strided_slice_1/pfor/concat_1/values_0:output:0:loop_body/rotation_matrix/strided_slice_1/stack_1:output:0Eloop_body/rotation_matrix/strided_slice_1/pfor/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
@loop_body/rotation_matrix/strided_slice_1/pfor/concat_2/values_0Const*
_output_shapes
:*
dtype0*
valueB:~
<loop_body/rotation_matrix/strided_slice_1/pfor/concat_2/axisConst*
_output_shapes
: *
dtype0*
value	B : ¿
7loop_body/rotation_matrix/strided_slice_1/pfor/concat_2ConcatV2Iloop_body/rotation_matrix/strided_slice_1/pfor/concat_2/values_0:output:0:loop_body/rotation_matrix/strided_slice_1/stack_2:output:0Eloop_body/rotation_matrix/strided_slice_1/pfor/concat_2/axis:output:0*
N*
T0*
_output_shapes
:·
;loop_body/rotation_matrix/strided_slice_1/pfor/StridedSliceStridedSlice,loop_body/rotation_matrix/Cos_2/pfor/Cos:y:0>loop_body/rotation_matrix/strided_slice_1/pfor/concat:output:0@loop_body/rotation_matrix/strided_slice_1/pfor/concat_1:output:0@loop_body/rotation_matrix/strided_slice_1/pfor/concat_2:output:0*
Index0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*

begin_mask*
end_mask*
new_axis_mask|
+loop_body/rotation_matrix/concat/pfor/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      
Eloop_body/rotation_matrix/concat/pfor/ones_like/Shape/shape_as_tensorConst*
_output_shapes
:*
dtype0*
valueB:w
5loop_body/rotation_matrix/concat/pfor/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :ì
/loop_body/rotation_matrix/concat/pfor/ones_likeFillNloop_body/rotation_matrix/concat/pfor/ones_like/Shape/shape_as_tensor:output:0>loop_body/rotation_matrix/concat/pfor/ones_like/Const:output:0*
T0*
_output_shapes
:
3loop_body/rotation_matrix/concat/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿÕ
-loop_body/rotation_matrix/concat/pfor/ReshapeReshape8loop_body/rotation_matrix/concat/pfor/ones_like:output:0<loop_body/rotation_matrix/concat/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:
5loop_body/rotation_matrix/concat/pfor/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿ¶
/loop_body/rotation_matrix/concat/pfor/Reshape_1Reshapepfor/Reshape:output:0>loop_body/rotation_matrix/concat/pfor/Reshape_1/shape:output:0*
T0*
_output_shapes
:s
1loop_body/rotation_matrix/concat/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
,loop_body/rotation_matrix/concat/pfor/concatConcatV28loop_body/rotation_matrix/concat/pfor/Reshape_1:output:06loop_body/rotation_matrix/concat/pfor/Reshape:output:0:loop_body/rotation_matrix/concat/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:v
4loop_body/rotation_matrix/concat/pfor/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B : Ô
0loop_body/rotation_matrix/concat/pfor/ExpandDims
ExpandDims(loop_body/rotation_matrix/zeros:output:0=loop_body/rotation_matrix/concat/pfor/ExpandDims/dim:output:0*
T0*"
_output_shapes
:Ú
*loop_body/rotation_matrix/concat/pfor/TileTile9loop_body/rotation_matrix/concat/pfor/ExpandDims:output:05loop_body/rotation_matrix/concat/pfor/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
4loop_body/rotation_matrix/concat/pfor/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : Ò
2loop_body/rotation_matrix/concat/pfor/GreaterEqualGreaterEqual.loop_body/rotation_matrix/concat/axis:output:0=loop_body/rotation_matrix/concat/pfor/GreaterEqual/y:output:0*
T0*
_output_shapes
: 
*loop_body/rotation_matrix/concat/pfor/CastCast6loop_body/rotation_matrix/concat/pfor/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
: ³
)loop_body/rotation_matrix/concat/pfor/addAddV2.loop_body/rotation_matrix/concat/axis:output:0.loop_body/rotation_matrix/concat/pfor/Cast:y:0*
T0*
_output_shapes
: ç
.loop_body/rotation_matrix/concat/pfor/concat_1ConcatV2Dloop_body/rotation_matrix/strided_slice_1/pfor/StridedSlice:output:0*loop_body/rotation_matrix/Neg/pfor/Neg:y:0Dloop_body/rotation_matrix/strided_slice_3/pfor/StridedSlice:output:0Dloop_body/rotation_matrix/strided_slice_4/pfor/StridedSlice:output:0Dloop_body/rotation_matrix/strided_slice_5/pfor/StridedSlice:output:0Dloop_body/rotation_matrix/strided_slice_6/pfor/StridedSlice:output:03loop_body/rotation_matrix/concat/pfor/Tile:output:0-loop_body/rotation_matrix/concat/pfor/add:z:0*
N*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
loop_body/SelectV2/pfor/RankConst*
_output_shapes
: *
dtype0*
value	B : _
loop_body/SelectV2/pfor/add/yConst*
_output_shapes
: *
dtype0*
value	B :
loop_body/SelectV2/pfor/addAddV2%loop_body/SelectV2/pfor/Rank:output:0&loop_body/SelectV2/pfor/add/y:output:0*
T0*
_output_shapes
: `
loop_body/SelectV2/pfor/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :`
loop_body/SelectV2/pfor/Rank_2Const*
_output_shapes
: *
dtype0*
value	B : a
loop_body/SelectV2/pfor/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
loop_body/SelectV2/pfor/add_1AddV2'loop_body/SelectV2/pfor/Rank_2:output:0(loop_body/SelectV2/pfor/add_1/y:output:0*
T0*
_output_shapes
: 
loop_body/SelectV2/pfor/MaximumMaximum'loop_body/SelectV2/pfor/Rank_1:output:0loop_body/SelectV2/pfor/add:z:0*
T0*
_output_shapes
: 
!loop_body/SelectV2/pfor/Maximum_1Maximum!loop_body/SelectV2/pfor/add_1:z:0#loop_body/SelectV2/pfor/Maximum:z:0*
T0*
_output_shapes
: `
loop_body/SelectV2/pfor/ShapeShapepfor/range:output:0*
T0*
_output_shapes
:
loop_body/SelectV2/pfor/subSub%loop_body/SelectV2/pfor/Maximum_1:z:0'loop_body/SelectV2/pfor/Rank_1:output:0*
T0*
_output_shapes
: o
%loop_body/SelectV2/pfor/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB: 
loop_body/SelectV2/pfor/ReshapeReshapeloop_body/SelectV2/pfor/sub:z:0.loop_body/SelectV2/pfor/Reshape/shape:output:0*
T0*
_output_shapes
:l
"loop_body/SelectV2/pfor/Tile/inputConst*
_output_shapes
:*
dtype0*
valueB:
loop_body/SelectV2/pfor/TileTile+loop_body/SelectV2/pfor/Tile/input:output:0(loop_body/SelectV2/pfor/Reshape:output:0*
T0*
_output_shapes
: u
+loop_body/SelectV2/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-loop_body/SelectV2/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-loop_body/SelectV2/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ç
%loop_body/SelectV2/pfor/strided_sliceStridedSlice&loop_body/SelectV2/pfor/Shape:output:04loop_body/SelectV2/pfor/strided_slice/stack:output:06loop_body/SelectV2/pfor/strided_slice/stack_1:output:06loop_body/SelectV2/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*

begin_maskw
-loop_body/SelectV2/pfor/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:y
/loop_body/SelectV2/pfor/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: y
/loop_body/SelectV2/pfor/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ë
'loop_body/SelectV2/pfor/strided_slice_1StridedSlice&loop_body/SelectV2/pfor/Shape:output:06loop_body/SelectV2/pfor/strided_slice_1/stack:output:08loop_body/SelectV2/pfor/strided_slice_1/stack_1:output:08loop_body/SelectV2/pfor/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maske
#loop_body/SelectV2/pfor/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 
loop_body/SelectV2/pfor/concatConcatV2.loop_body/SelectV2/pfor/strided_slice:output:0%loop_body/SelectV2/pfor/Tile:output:00loop_body/SelectV2/pfor/strided_slice_1:output:0,loop_body/SelectV2/pfor/concat/axis:output:0*
N*
T0*
_output_shapes
:
!loop_body/SelectV2/pfor/Reshape_1Reshapepfor/range:output:0'loop_body/SelectV2/pfor/concat:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
 loop_body/SelectV2/pfor/SelectV2SelectV2loop_body/Greater:z:0*loop_body/SelectV2/pfor/Reshape_1:output:0loop_body/SelectV2/e:output:0*
T0*#
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
%loop_body/GatherV2/pfor/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : î
 loop_body/GatherV2/pfor/GatherV2GatherV2inputs)loop_body/SelectV2/pfor/SelectV2:output:0.loop_body/GatherV2/pfor/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
(loop_body/ExpandDims/pfor/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
value	B : ­
&loop_body/ExpandDims/pfor/GreaterEqualGreaterEqual!loop_body/ExpandDims/dim:output:01loop_body/ExpandDims/pfor/GreaterEqual/y:output:0*
T0*
_output_shapes
: 
loop_body/ExpandDims/pfor/CastCast*loop_body/ExpandDims/pfor/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
: 
loop_body/ExpandDims/pfor/addAddV2!loop_body/ExpandDims/dim:output:0"loop_body/ExpandDims/pfor/Cast:y:0*
T0*
_output_shapes
: À
$loop_body/ExpandDims/pfor/ExpandDims
ExpandDims)loop_body/GatherV2/pfor/GatherV2:output:0!loop_body/ExpandDims/pfor/add:z:0*
T0*5
_output_shapes#
!:ÿÿÿÿÿÿÿÿÿ
Gloop_body/transform/ImageProjectiveTransformV3/pfor/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Iloop_body/transform/ImageProjectiveTransformV3/pfor/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Iloop_body/transform/ImageProjectiveTransformV3/pfor/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¨
Aloop_body/transform/ImageProjectiveTransformV3/pfor/strided_sliceStridedSlicepfor/Reshape:output:0Ploop_body/transform/ImageProjectiveTransformV3/pfor/strided_slice/stack:output:0Rloop_body/transform/ImageProjectiveTransformV3/pfor/strided_slice/stack_1:output:0Rloop_body/transform/ImageProjectiveTransformV3/pfor/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
Oloop_body/transform/ImageProjectiveTransformV3/pfor/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÎ
Aloop_body/transform/ImageProjectiveTransformV3/pfor/TensorArrayV2TensorListReserveXloop_body/transform/ImageProjectiveTransformV3/pfor/TensorArrayV2/element_shape:output:0Jloop_body/transform/ImageProjectiveTransformV3/pfor/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ{
9loop_body/transform/ImageProjectiveTransformV3/pfor/ConstConst*
_output_shapes
: *
dtype0*
value	B : 
Lloop_body/transform/ImageProjectiveTransformV3/pfor/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ
Floop_body/transform/ImageProjectiveTransformV3/pfor/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : þ
9loop_body/transform/ImageProjectiveTransformV3/pfor/whileStatelessWhileOloop_body/transform/ImageProjectiveTransformV3/pfor/while/loop_counter:output:0Uloop_body/transform/ImageProjectiveTransformV3/pfor/while/maximum_iterations:output:0Bloop_body/transform/ImageProjectiveTransformV3/pfor/Const:output:0Jloop_body/transform/ImageProjectiveTransformV3/pfor/TensorArrayV2:handle:0Jloop_body/transform/ImageProjectiveTransformV3/pfor/strided_slice:output:0-loop_body/ExpandDims/pfor/ExpandDims:output:07loop_body/rotation_matrix/concat/pfor/concat_1:output:0*loop_body/transform/strided_slice:output:0'loop_body/transform/fill_value:output:0*
T
2	*
_lower_using_switch_merge(*
_num_original_outputs	*^
_output_shapesL
J: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: * 
_read_only_resource_inputs
 *
_stateful_parallelism( *O
bodyGRE
Cloop_body_transform_ImageProjectiveTransformV3_pfor_while_body_8725*O
condGRE
Cloop_body_transform_ImageProjectiveTransformV3_pfor_while_cond_8724*]
output_shapesL
J: : : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:: ~
;loop_body/transform/ImageProjectiveTransformV3/pfor/Const_1Const*
_output_shapes
: *
dtype0	*
valueB	 ±
Tloop_body/transform/ImageProjectiveTransformV3/pfor/TensorListConcatV2/element_shapeConst*
_output_shapes
:*
dtype0*)
value B"ÿÿÿÿ   ÿÿÿÿÿÿÿÿ   ²
Floop_body/transform/ImageProjectiveTransformV3/pfor/TensorListConcatV2TensorListConcatV2Bloop_body/transform/ImageProjectiveTransformV3/pfor/while:output:3]loop_body/transform/ImageProjectiveTransformV3/pfor/TensorListConcatV2/element_shape:output:0Dloop_body/transform/ImageProjectiveTransformV3/pfor/Const_1:output:0*D
_output_shapes2
0:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*
element_dtype0*

shape_type0Í
loop_body/Squeeze/pfor/SqueezeSqueezeOloop_body/transform/ImageProjectiveTransformV3/pfor/TensorListConcatV2:tensor:0*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims

IdentityIdentity'loop_body/Squeeze/pfor/Squeeze:output:0^NoOp*
T0*1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©
NoOpNoOp*^loop_body/stateful_uniform/RngReadAndSkip5^loop_body/stateful_uniform/RngReadAndSkip/pfor/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: 2V
)loop_body/stateful_uniform/RngReadAndSkip)loop_body/stateful_uniform/RngReadAndSkip2l
4loop_body/stateful_uniform/RngReadAndSkip/pfor/while4loop_body/stateful_uniform/RngReadAndSkip/pfor/while:Y U
1
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ý
©
Irandom_rotation_loop_body_stateful_uniform_Bitcast_1_pfor_while_cond_7120
random_rotation_loop_body_stateful_uniform_bitcast_1_pfor_while_random_rotation_loop_body_stateful_uniform_bitcast_1_pfor_while_loop_counter
random_rotation_loop_body_stateful_uniform_bitcast_1_pfor_while_random_rotation_loop_body_stateful_uniform_bitcast_1_pfor_while_maximum_iterationsO
Krandom_rotation_loop_body_stateful_uniform_bitcast_1_pfor_while_placeholderQ
Mrandom_rotation_loop_body_stateful_uniform_bitcast_1_pfor_while_placeholder_1
random_rotation_loop_body_stateful_uniform_bitcast_1_pfor_while_less_random_rotation_loop_body_stateful_uniform_bitcast_1_pfor_strided_slice§
¢random_rotation_loop_body_stateful_uniform_bitcast_1_pfor_while_random_rotation_loop_body_stateful_uniform_bitcast_1_pfor_while_cond_7120___redundant_placeholder0	L
Hrandom_rotation_loop_body_stateful_uniform_bitcast_1_pfor_while_identity
É
Drandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/while/LessLessKrandom_rotation_loop_body_stateful_uniform_bitcast_1_pfor_while_placeholderrandom_rotation_loop_body_stateful_uniform_bitcast_1_pfor_while_less_random_rotation_loop_body_stateful_uniform_bitcast_1_pfor_strided_slice*
T0*
_output_shapes
: ¿
Hrandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/while/IdentityIdentityHrandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/while/Less:z:0*
T0
*
_output_shapes
: "
Hrandom_rotation_loop_body_stateful_uniform_bitcast_1_pfor_while_identityQrandom_rotation/loop_body/stateful_uniform/Bitcast_1/pfor/while/Identity:output:0*(
_construction_contextkEagerRuntime*!
_input_shapes
: : : : : :: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
:"µ	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Æ
serving_default²
Y
random_flip_inputD
#serving_default_random_flip_input:0ÿÿÿÿÿÿÿÿÿ9
dense0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:±
Þ
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer-4
layer_with_weights-1
layer-5
layer-6
layer_with_weights-2
layer-7
	layer-8

layer-9
layer_with_weights-3
layer-10
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
¼
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator"
_tf_keras_layer
¼
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses
"_random_generator"
_tf_keras_layer
¥
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses"
_tf_keras_layer
Ý
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses

/kernel
0bias
 1_jit_compiled_convolution_op"
_tf_keras_layer
¥
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses"
_tf_keras_layer
Ý
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses

>kernel
?bias
 @_jit_compiled_convolution_op"
_tf_keras_layer
¥
A	variables
Btrainable_variables
Cregularization_losses
D	keras_api
E__call__
*F&call_and_return_all_conditional_losses"
_tf_keras_layer
Ý
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses

Mkernel
Nbias
 O_jit_compiled_convolution_op"
_tf_keras_layer
¥
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses"
_tf_keras_layer
¼
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
Z__call__
*[&call_and_return_all_conditional_losses
\_random_generator"
_tf_keras_layer
»
]	variables
^trainable_variables
_regularization_losses
`	keras_api
a__call__
*b&call_and_return_all_conditional_losses

ckernel
dbias"
_tf_keras_layer
X
/0
01
>2
?3
M4
N5
c6
d7"
trackable_list_wrapper
X
/0
01
>2
?3
M4
N5
c6
d7"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
enon_trainable_variables

flayers
gmetrics
hlayer_regularization_losses
ilayer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ù
jtrace_0
ktrace_1
ltrace_2
mtrace_32î
)__inference_sequential_layer_call_fn_5428
)__inference_sequential_layer_call_fn_6712
)__inference_sequential_layer_call_fn_6735
)__inference_sequential_layer_call_fn_6598¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zjtrace_0zktrace_1zltrace_2zmtrace_3
Å
ntrace_0
otrace_1
ptrace_2
qtrace_32Ú
D__inference_sequential_layer_call_and_return_conditional_losses_6777
D__inference_sequential_layer_call_and_return_conditional_losses_7792
D__inference_sequential_layer_call_and_return_conditional_losses_6629
D__inference_sequential_layer_call_and_return_conditional_losses_6662¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zntrace_0zotrace_1zptrace_2zqtrace_3
ÔBÑ
__inference__wrapped_model_5269random_flip_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó
riter

sbeta_1

tbeta_2
	udecay
vlearning_rate/mÚ0mÛ>mÜ?mÝMmÞNmßcmàdmá/vâ0vã>vä?våMvæNvçcvèdvé"
	optimizer
,
wserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
xnon_trainable_variables

ylayers
zmetrics
{layer_regularization_losses
|layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Å
}trace_0
~trace_12
*__inference_random_flip_layer_call_fn_7797
*__inference_random_flip_layer_call_fn_7802³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults¢
p

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z}trace_0z~trace_1
ý
trace_0
trace_12Ä
E__inference_random_flip_layer_call_and_return_conditional_losses_7806
E__inference_random_flip_layer_call_and_return_conditional_losses_7865³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults¢
p

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1
/

_generator"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
Ñ
trace_0
trace_12
.__inference_random_rotation_layer_call_fn_7870
.__inference_random_rotation_layer_call_fn_7877³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults¢
p

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1

trace_0
trace_12Ì
I__inference_random_rotation_layer_call_and_return_conditional_losses_7881
I__inference_random_rotation_layer_call_and_return_conditional_losses_8793³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults¢
p

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1
/

_generator"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
î
trace_02Ï
(__inference_rescaling_layer_call_fn_8798¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02ê
C__inference_rescaling_layer_call_and_return_conditional_losses_8807¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
ë
trace_02Ì
%__inference_conv2d_layer_call_fn_8816¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02ç
@__inference_conv2d_layer_call_and_return_conditional_losses_8827¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
':%2conv2d/kernel
:2conv2d/bias
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
ò
trace_02Ó
,__inference_max_pooling2d_layer_call_fn_8832¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

 trace_02î
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_8837¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z trace_0
.
>0
?1"
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¡non_trainable_variables
¢layers
£metrics
 ¤layer_regularization_losses
¥layer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
í
¦trace_02Î
'__inference_conv2d_1_layer_call_fn_8846¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¦trace_0

§trace_02é
B__inference_conv2d_1_layer_call_and_return_conditional_losses_8857¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z§trace_0
):'02conv2d_1/kernel
:02conv2d_1/bias
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
¨non_trainable_variables
©layers
ªmetrics
 «layer_regularization_losses
¬layer_metrics
A	variables
Btrainable_variables
Cregularization_losses
E__call__
*F&call_and_return_all_conditional_losses
&F"call_and_return_conditional_losses"
_generic_user_object
ô
­trace_02Õ
.__inference_max_pooling2d_1_layer_call_fn_8862¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z­trace_0

®trace_02ð
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_8867¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z®trace_0
.
M0
N1"
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
¯non_trainable_variables
°layers
±metrics
 ²layer_regularization_losses
³layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
í
´trace_02Î
'__inference_conv2d_2_layer_call_fn_8876¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z´trace_0

µtrace_02é
B__inference_conv2d_2_layer_call_and_return_conditional_losses_8887¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zµtrace_0
):'0`2conv2d_2/kernel
:`2conv2d_2/bias
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
¶non_trainable_variables
·layers
¸metrics
 ¹layer_regularization_losses
ºlayer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
ì
»trace_02Í
&__inference_flatten_layer_call_fn_8892¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z»trace_0

¼trace_02è
A__inference_flatten_layer_call_and_return_conditional_losses_8898¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¼trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
½non_trainable_variables
¾layers
¿metrics
 Àlayer_regularization_losses
Álayer_metrics
V	variables
Wtrainable_variables
Xregularization_losses
Z__call__
*[&call_and_return_all_conditional_losses
&["call_and_return_conditional_losses"
_generic_user_object
Á
Âtrace_0
Ãtrace_12
&__inference_dropout_layer_call_fn_8903
&__inference_dropout_layer_call_fn_8908³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÂtrace_0zÃtrace_1
÷
Ätrace_0
Åtrace_12¼
A__inference_dropout_layer_call_and_return_conditional_losses_8913
A__inference_dropout_layer_call_and_return_conditional_losses_8925³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÄtrace_0zÅtrace_1
"
_generic_user_object
.
c0
d1"
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ænon_trainable_variables
Çlayers
Èmetrics
 Élayer_regularization_losses
Êlayer_metrics
]	variables
^trainable_variables
_regularization_losses
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
ê
Ëtrace_02Ë
$__inference_dense_layer_call_fn_8934¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zËtrace_0

Ìtrace_02æ
?__inference_dense_layer_call_and_return_conditional_losses_8945¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÌtrace_0
 :
à2dense/kernel
:2
dense/bias
 "
trackable_list_wrapper
n
0
1
2
3
4
5
6
7
	8

9
10"
trackable_list_wrapper
0
Í0
Î1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
)__inference_sequential_layer_call_fn_5428random_flip_input"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
úB÷
)__inference_sequential_layer_call_fn_6712inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
úB÷
)__inference_sequential_layer_call_fn_6735inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
)__inference_sequential_layer_call_fn_6598random_flip_input"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
D__inference_sequential_layer_call_and_return_conditional_losses_6777inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
D__inference_sequential_layer_call_and_return_conditional_losses_7792inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 B
D__inference_sequential_layer_call_and_return_conditional_losses_6629random_flip_input"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 B
D__inference_sequential_layer_call_and_return_conditional_losses_6662random_flip_input"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ÓBÐ
"__inference_signature_wrapper_6691random_flip_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ïBì
*__inference_random_flip_layer_call_fn_7797inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults¢
p

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ïBì
*__inference_random_flip_layer_call_fn_7802inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults¢
p

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
E__inference_random_flip_layer_call_and_return_conditional_losses_7806inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults¢
p

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
E__inference_random_flip_layer_call_and_return_conditional_losses_7865inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults¢
p

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
/
Ï
_state_var"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
óBð
.__inference_random_rotation_layer_call_fn_7870inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults¢
p

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
óBð
.__inference_random_rotation_layer_call_fn_7877inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults¢
p

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
I__inference_random_rotation_layer_call_and_return_conditional_losses_7881inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults¢
p

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
I__inference_random_rotation_layer_call_and_return_conditional_losses_8793inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults¢
p

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
/
Ð
_state_var"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÜBÙ
(__inference_rescaling_layer_call_fn_8798inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷Bô
C__inference_rescaling_layer_call_and_return_conditional_losses_8807inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÙBÖ
%__inference_conv2d_layer_call_fn_8816inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ôBñ
@__inference_conv2d_layer_call_and_return_conditional_losses_8827inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
àBÝ
,__inference_max_pooling2d_layer_call_fn_8832inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ûBø
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_8837inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÛBØ
'__inference_conv2d_1_layer_call_fn_8846inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
öBó
B__inference_conv2d_1_layer_call_and_return_conditional_losses_8857inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
âBß
.__inference_max_pooling2d_1_layer_call_fn_8862inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ýBú
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_8867inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÛBØ
'__inference_conv2d_2_layer_call_fn_8876inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
öBó
B__inference_conv2d_2_layer_call_and_return_conditional_losses_8887inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÚB×
&__inference_flatten_layer_call_fn_8892inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
õBò
A__inference_flatten_layer_call_and_return_conditional_losses_8898inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ëBè
&__inference_dropout_layer_call_fn_8903inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ëBè
&__inference_dropout_layer_call_fn_8908inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
A__inference_dropout_layer_call_and_return_conditional_losses_8913inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
A__inference_dropout_layer_call_and_return_conditional_losses_8925inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ØBÕ
$__inference_dense_layer_call_fn_8934inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
óBð
?__inference_dense_layer_call_and_return_conditional_losses_8945inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
R
Ñ	variables
Ò	keras_api

Ótotal

Ôcount"
_tf_keras_metric
c
Õ	variables
Ö	keras_api

×total

Øcount
Ù
_fn_kwargs"
_tf_keras_metric
 :	2random_flip/StateVar
$:"	2random_rotation/StateVar
0
Ó0
Ô1"
trackable_list_wrapper
.
Ñ	variables"
_generic_user_object
:  (2total
:  (2count
0
×0
Ø1"
trackable_list_wrapper
.
Õ	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
,:*2Adam/conv2d/kernel/m
:2Adam/conv2d/bias/m
.:,02Adam/conv2d_1/kernel/m
 :02Adam/conv2d_1/bias/m
.:,0`2Adam/conv2d_2/kernel/m
 :`2Adam/conv2d_2/bias/m
%:#
à2Adam/dense/kernel/m
:2Adam/dense/bias/m
,:*2Adam/conv2d/kernel/v
:2Adam/conv2d/bias/v
.:,02Adam/conv2d_1/kernel/v
 :02Adam/conv2d_1/bias/v
.:,0`2Adam/conv2d_2/kernel/v
 :`2Adam/conv2d_2/bias/v
%:#
à2Adam/dense/kernel/v
:2Adam/dense/bias/v¢
__inference__wrapped_model_5269/0>?MNcdD¢A
:¢7
52
random_flip_inputÿÿÿÿÿÿÿÿÿ
ª "-ª*
(
dense
denseÿÿÿÿÿÿÿÿÿ²
B__inference_conv2d_1_layer_call_and_return_conditional_losses_8857l>?7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ~~
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿzz0
 
'__inference_conv2d_1_layer_call_fn_8846_>?7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ~~
ª " ÿÿÿÿÿÿÿÿÿzz0²
B__inference_conv2d_2_layer_call_and_return_conditional_losses_8887lMN7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ==0
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ99`
 
'__inference_conv2d_2_layer_call_fn_8876_MN7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ==0
ª " ÿÿÿÿÿÿÿÿÿ99`´
@__inference_conv2d_layer_call_and_return_conditional_losses_8827p/09¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿüü
 
%__inference_conv2d_layer_call_fn_8816c/09¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ
ª ""ÿÿÿÿÿÿÿÿÿüü¡
?__inference_dense_layer_call_and_return_conditional_losses_8945^cd1¢.
'¢$
"
inputsÿÿÿÿÿÿÿÿÿà
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 y
$__inference_dense_layer_call_fn_8934Qcd1¢.
'¢$
"
inputsÿÿÿÿÿÿÿÿÿà
ª "ÿÿÿÿÿÿÿÿÿ¥
A__inference_dropout_layer_call_and_return_conditional_losses_8913`5¢2
+¢(
"
inputsÿÿÿÿÿÿÿÿÿà
p 
ª "'¢$

0ÿÿÿÿÿÿÿÿÿà
 ¥
A__inference_dropout_layer_call_and_return_conditional_losses_8925`5¢2
+¢(
"
inputsÿÿÿÿÿÿÿÿÿà
p
ª "'¢$

0ÿÿÿÿÿÿÿÿÿà
 }
&__inference_dropout_layer_call_fn_8903S5¢2
+¢(
"
inputsÿÿÿÿÿÿÿÿÿà
p 
ª "ÿÿÿÿÿÿÿÿÿà}
&__inference_dropout_layer_call_fn_8908S5¢2
+¢(
"
inputsÿÿÿÿÿÿÿÿÿà
p
ª "ÿÿÿÿÿÿÿÿÿà§
A__inference_flatten_layer_call_and_return_conditional_losses_8898b7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ99`
ª "'¢$

0ÿÿÿÿÿÿÿÿÿà
 
&__inference_flatten_layer_call_fn_8892U7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ99`
ª "ÿÿÿÿÿÿÿÿÿàì
I__inference_max_pooling2d_1_layer_call_and_return_conditional_losses_8867R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Ä
.__inference_max_pooling2d_1_layer_call_fn_8862R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿê
G__inference_max_pooling2d_layer_call_and_return_conditional_losses_8837R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 Â
,__inference_max_pooling2d_layer_call_fn_8832R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¹
E__inference_random_flip_layer_call_and_return_conditional_losses_7806p=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 ¹
E__inference_random_flip_layer_call_and_return_conditional_losses_7865p=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 
*__inference_random_flip_layer_call_fn_7797c=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p 
ª ""ÿÿÿÿÿÿÿÿÿ
*__inference_random_flip_layer_call_fn_7802c=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p
ª ""ÿÿÿÿÿÿÿÿÿ½
I__inference_random_rotation_layer_call_and_return_conditional_losses_7881p=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p 
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 Á
I__inference_random_rotation_layer_call_and_return_conditional_losses_8793tÐ=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 
.__inference_random_rotation_layer_call_fn_7870c=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p 
ª ""ÿÿÿÿÿÿÿÿÿ
.__inference_random_rotation_layer_call_fn_7877gÐ=¢:
3¢0
*'
inputsÿÿÿÿÿÿÿÿÿ
p
ª ""ÿÿÿÿÿÿÿÿÿ³
C__inference_rescaling_layer_call_and_return_conditional_losses_8807l9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ
ª "/¢,
%"
0ÿÿÿÿÿÿÿÿÿ
 
(__inference_rescaling_layer_call_fn_8798_9¢6
/¢,
*'
inputsÿÿÿÿÿÿÿÿÿ
ª ""ÿÿÿÿÿÿÿÿÿÇ
D__inference_sequential_layer_call_and_return_conditional_losses_6629/0>?MNcdL¢I
B¢?
52
random_flip_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ê
D__inference_sequential_layer_call_and_return_conditional_losses_6662
Ð/0>?MNcdL¢I
B¢?
52
random_flip_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¼
D__inference_sequential_layer_call_and_return_conditional_losses_6777t/0>?MNcdA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¾
D__inference_sequential_layer_call_and_return_conditional_losses_7792v
Ð/0>?MNcdA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
)__inference_sequential_layer_call_fn_5428r/0>?MNcdL¢I
B¢?
52
random_flip_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ¡
)__inference_sequential_layer_call_fn_6598t
Ð/0>?MNcdL¢I
B¢?
52
random_flip_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
)__inference_sequential_layer_call_fn_6712g/0>?MNcdA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
)__inference_sequential_layer_call_fn_6735i
Ð/0>?MNcdA¢>
7¢4
*'
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ»
"__inference_signature_wrapper_6691/0>?MNcdY¢V
¢ 
OªL
J
random_flip_input52
random_flip_inputÿÿÿÿÿÿÿÿÿ"-ª*
(
dense
denseÿÿÿÿÿÿÿÿÿ