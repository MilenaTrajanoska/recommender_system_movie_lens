??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
l
BatchMatMulV2
x"T
y"T
output"T"
Ttype:
2		"
adj_xbool( "
adj_ybool( 
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
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
.
Identity

input"T
output"T"	
Ttype
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
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
dtypetype?
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
?
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
executor_typestring ?
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
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718??
?
MoviesEmbedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
?K?*+
shared_nameMoviesEmbedding/embeddings
?
.MoviesEmbedding/embeddings/Read/ReadVariableOpReadVariableOpMoviesEmbedding/embeddings* 
_output_shapes
:
?K?*
dtype0
?
UsersEmbedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??**
shared_nameUsersEmbedding/embeddings
?
-UsersEmbedding/embeddings/Read/ReadVariableOpReadVariableOpUsersEmbedding/embeddings* 
_output_shapes
:
??*
dtype0
d
SGD/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name
SGD/iter
]
SGD/iter/Read/ReadVariableOpReadVariableOpSGD/iter*
_output_shapes
: *
dtype0	
f
	SGD/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	SGD/decay
_
SGD/decay/Read/ReadVariableOpReadVariableOp	SGD/decay*
_output_shapes
: *
dtype0
v
SGD/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameSGD/learning_rate
o
%SGD/learning_rate/Read/ReadVariableOpReadVariableOpSGD/learning_rate*
_output_shapes
: *
dtype0
l
SGD/momentumVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameSGD/momentum
e
 SGD/momentum/Read/ReadVariableOpReadVariableOpSGD/momentum*
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
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
	optimizer
	regularization_losses

trainable_variables
	variables
	keras_api

signatures

_init_input_shape

_init_input_shape
b

embeddings
regularization_losses
trainable_variables
	variables
	keras_api
b

embeddings
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
 	variables
!	keras_api
R
"regularization_losses
#trainable_variables
$	variables
%	keras_api
6
&iter
	'decay
(learning_rate
)momentum
 

0
1

0
1
?
	regularization_losses

trainable_variables
*layer_metrics

+layers
	variables
,layer_regularization_losses
-non_trainable_variables
.metrics
 
 
 
jh
VARIABLE_VALUEMoviesEmbedding/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
?
regularization_losses
trainable_variables
/layer_metrics

0layers
	variables
1layer_regularization_losses
2non_trainable_variables
3metrics
ig
VARIABLE_VALUEUsersEmbedding/embeddings:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
?
regularization_losses
trainable_variables
4layer_metrics

5layers
	variables
6layer_regularization_losses
7non_trainable_variables
8metrics
 
 
 
?
regularization_losses
trainable_variables
9layer_metrics

:layers
	variables
;layer_regularization_losses
<non_trainable_variables
=metrics
 
 
 
?
regularization_losses
trainable_variables
>layer_metrics

?layers
 	variables
@layer_regularization_losses
Anon_trainable_variables
Bmetrics
 
 
 
?
"regularization_losses
#trainable_variables
Clayer_metrics

Dlayers
$	variables
Elayer_regularization_losses
Fnon_trainable_variables
Gmetrics
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUESGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
 
1
0
1
2
3
4
5
6
 
 

H0
I1
J2
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	Ktotal
	Lcount
M	variables
N	keras_api
D
	Ototal
	Pcount
Q
_fn_kwargs
R	variables
S	keras_api
D
	Ttotal
	Ucount
V
_fn_kwargs
W	variables
X	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

K0
L1

M	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

O0
P1

R	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

T0
U1

W	variables
~
serving_default_MoviesInputPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
}
serving_default_UsersInputPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_MoviesInputserving_default_UsersInputUsersEmbedding/embeddingsMoviesEmbedding/embeddings*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_3050674
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename.MoviesEmbedding/embeddings/Read/ReadVariableOp-UsersEmbedding/embeddings/Read/ReadVariableOpSGD/iter/Read/ReadVariableOpSGD/decay/Read/ReadVariableOp%SGD/learning_rate/Read/ReadVariableOp SGD/momentum/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOpConst*
Tin
2	*
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
GPU 2J 8? *)
f$R"
 __inference__traced_save_3050954
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameMoviesEmbedding/embeddingsUsersEmbedding/embeddingsSGD/iter	SGD/decaySGD/learning_rateSGD/momentumtotalcounttotal_1count_1total_2count_2*
Tin
2*
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
GPU 2J 8? *,
f'R%
#__inference__traced_restore_3051000??
?;
?
[__inference_MatrixFactorizationReccomender_layer_call_and_return_conditional_losses_3050714
inputs_0
inputs_1;
'usersembedding_embedding_lookup_3050679:
??<
(moviesembedding_embedding_lookup_3050685:
?K?
identity?? MoviesEmbedding/embedding_lookup?<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp?UsersEmbedding/embedding_lookup?;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp}
UsersEmbedding/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
UsersEmbedding/Cast?
UsersEmbedding/embedding_lookupResourceGather'usersembedding_embedding_lookup_3050679UsersEmbedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*:
_class0
.,loc:@UsersEmbedding/embedding_lookup/3050679*,
_output_shapes
:??????????*
dtype02!
UsersEmbedding/embedding_lookup?
(UsersEmbedding/embedding_lookup/IdentityIdentity(UsersEmbedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*:
_class0
.,loc:@UsersEmbedding/embedding_lookup/3050679*,
_output_shapes
:??????????2*
(UsersEmbedding/embedding_lookup/Identity?
*UsersEmbedding/embedding_lookup/Identity_1Identity1UsersEmbedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2,
*UsersEmbedding/embedding_lookup/Identity_1
MoviesEmbedding/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:?????????2
MoviesEmbedding/Cast?
 MoviesEmbedding/embedding_lookupResourceGather(moviesembedding_embedding_lookup_3050685MoviesEmbedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*;
_class1
/-loc:@MoviesEmbedding/embedding_lookup/3050685*,
_output_shapes
:??????????*
dtype02"
 MoviesEmbedding/embedding_lookup?
)MoviesEmbedding/embedding_lookup/IdentityIdentity)MoviesEmbedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*;
_class1
/-loc:@MoviesEmbedding/embedding_lookup/3050685*,
_output_shapes
:??????????2+
)MoviesEmbedding/embedding_lookup/Identity?
+MoviesEmbedding/embedding_lookup/Identity_1Identity2MoviesEmbedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2-
+MoviesEmbedding/embedding_lookup/Identity_1{
MoviesFlatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????,  2
MoviesFlatten/Const?
MoviesFlatten/ReshapeReshape4MoviesEmbedding/embedding_lookup/Identity_1:output:0MoviesFlatten/Const:output:0*
T0*(
_output_shapes
:??????????2
MoviesFlatten/Reshapey
UsersFlatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????,  2
UsersFlatten/Const?
UsersFlatten/ReshapeReshape3UsersEmbedding/embedding_lookup/Identity_1:output:0UsersFlatten/Const:output:0*
T0*(
_output_shapes
:??????????2
UsersFlatten/Reshapex
DotProduct/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
DotProduct/ExpandDims/dim?
DotProduct/ExpandDims
ExpandDimsMoviesFlatten/Reshape:output:0"DotProduct/ExpandDims/dim:output:0*
T0*,
_output_shapes
:??????????2
DotProduct/ExpandDims|
DotProduct/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2
DotProduct/ExpandDims_1/dim?
DotProduct/ExpandDims_1
ExpandDimsUsersFlatten/Reshape:output:0$DotProduct/ExpandDims_1/dim:output:0*
T0*,
_output_shapes
:??????????2
DotProduct/ExpandDims_1?
DotProduct/MatMulBatchMatMulV2DotProduct/ExpandDims:output:0 DotProduct/ExpandDims_1:output:0*
T0*+
_output_shapes
:?????????2
DotProduct/MatMuln
DotProduct/ShapeShapeDotProduct/MatMul:output:0*
T0*
_output_shapes
:2
DotProduct/Shape?
DotProduct/SqueezeSqueezeDotProduct/MatMul:output:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
2
DotProduct/Squeeze?
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp(moviesembedding_embedding_lookup_3050685* 
_output_shapes
:
?K?*
dtype02>
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp?
-MoviesEmbedding/embeddings/Regularizer/SquareSquareDMoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
?K?2/
-MoviesEmbedding/embeddings/Regularizer/Square?
,MoviesEmbedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,MoviesEmbedding/embeddings/Regularizer/Const?
*MoviesEmbedding/embeddings/Regularizer/SumSum1MoviesEmbedding/embeddings/Regularizer/Square:y:05MoviesEmbedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*MoviesEmbedding/embeddings/Regularizer/Sum?
,MoviesEmbedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52.
,MoviesEmbedding/embeddings/Regularizer/mul/x?
*MoviesEmbedding/embeddings/Regularizer/mulMul5MoviesEmbedding/embeddings/Regularizer/mul/x:output:03MoviesEmbedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*MoviesEmbedding/embeddings/Regularizer/mul?
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp'usersembedding_embedding_lookup_3050679* 
_output_shapes
:
??*
dtype02=
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp?
,UsersEmbedding/embeddings/Regularizer/SquareSquareCUsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2.
,UsersEmbedding/embeddings/Regularizer/Square?
+UsersEmbedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+UsersEmbedding/embeddings/Regularizer/Const?
)UsersEmbedding/embeddings/Regularizer/SumSum0UsersEmbedding/embeddings/Regularizer/Square:y:04UsersEmbedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)UsersEmbedding/embeddings/Regularizer/Sum?
+UsersEmbedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52-
+UsersEmbedding/embeddings/Regularizer/mul/x?
)UsersEmbedding/embeddings/Regularizer/mulMul4UsersEmbedding/embeddings/Regularizer/mul/x:output:02UsersEmbedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)UsersEmbedding/embeddings/Regularizer/mul?
IdentityIdentityDotProduct/Squeeze:output:0!^MoviesEmbedding/embedding_lookup=^MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp ^UsersEmbedding/embedding_lookup<^UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:?????????:?????????: : 2D
 MoviesEmbedding/embedding_lookup MoviesEmbedding/embedding_lookup2|
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp2B
UsersEmbedding/embedding_lookupUsersEmbedding/embedding_lookup2z
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?;
?
[__inference_MatrixFactorizationReccomender_layer_call_and_return_conditional_losses_3050754
inputs_0
inputs_1;
'usersembedding_embedding_lookup_3050719:
??<
(moviesembedding_embedding_lookup_3050725:
?K?
identity?? MoviesEmbedding/embedding_lookup?<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp?UsersEmbedding/embedding_lookup?;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp}
UsersEmbedding/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
UsersEmbedding/Cast?
UsersEmbedding/embedding_lookupResourceGather'usersembedding_embedding_lookup_3050719UsersEmbedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*:
_class0
.,loc:@UsersEmbedding/embedding_lookup/3050719*,
_output_shapes
:??????????*
dtype02!
UsersEmbedding/embedding_lookup?
(UsersEmbedding/embedding_lookup/IdentityIdentity(UsersEmbedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*:
_class0
.,loc:@UsersEmbedding/embedding_lookup/3050719*,
_output_shapes
:??????????2*
(UsersEmbedding/embedding_lookup/Identity?
*UsersEmbedding/embedding_lookup/Identity_1Identity1UsersEmbedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2,
*UsersEmbedding/embedding_lookup/Identity_1
MoviesEmbedding/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:?????????2
MoviesEmbedding/Cast?
 MoviesEmbedding/embedding_lookupResourceGather(moviesembedding_embedding_lookup_3050725MoviesEmbedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*;
_class1
/-loc:@MoviesEmbedding/embedding_lookup/3050725*,
_output_shapes
:??????????*
dtype02"
 MoviesEmbedding/embedding_lookup?
)MoviesEmbedding/embedding_lookup/IdentityIdentity)MoviesEmbedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*;
_class1
/-loc:@MoviesEmbedding/embedding_lookup/3050725*,
_output_shapes
:??????????2+
)MoviesEmbedding/embedding_lookup/Identity?
+MoviesEmbedding/embedding_lookup/Identity_1Identity2MoviesEmbedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2-
+MoviesEmbedding/embedding_lookup/Identity_1{
MoviesFlatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????,  2
MoviesFlatten/Const?
MoviesFlatten/ReshapeReshape4MoviesEmbedding/embedding_lookup/Identity_1:output:0MoviesFlatten/Const:output:0*
T0*(
_output_shapes
:??????????2
MoviesFlatten/Reshapey
UsersFlatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????,  2
UsersFlatten/Const?
UsersFlatten/ReshapeReshape3UsersEmbedding/embedding_lookup/Identity_1:output:0UsersFlatten/Const:output:0*
T0*(
_output_shapes
:??????????2
UsersFlatten/Reshapex
DotProduct/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
DotProduct/ExpandDims/dim?
DotProduct/ExpandDims
ExpandDimsMoviesFlatten/Reshape:output:0"DotProduct/ExpandDims/dim:output:0*
T0*,
_output_shapes
:??????????2
DotProduct/ExpandDims|
DotProduct/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2
DotProduct/ExpandDims_1/dim?
DotProduct/ExpandDims_1
ExpandDimsUsersFlatten/Reshape:output:0$DotProduct/ExpandDims_1/dim:output:0*
T0*,
_output_shapes
:??????????2
DotProduct/ExpandDims_1?
DotProduct/MatMulBatchMatMulV2DotProduct/ExpandDims:output:0 DotProduct/ExpandDims_1:output:0*
T0*+
_output_shapes
:?????????2
DotProduct/MatMuln
DotProduct/ShapeShapeDotProduct/MatMul:output:0*
T0*
_output_shapes
:2
DotProduct/Shape?
DotProduct/SqueezeSqueezeDotProduct/MatMul:output:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
2
DotProduct/Squeeze?
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp(moviesembedding_embedding_lookup_3050725* 
_output_shapes
:
?K?*
dtype02>
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp?
-MoviesEmbedding/embeddings/Regularizer/SquareSquareDMoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
?K?2/
-MoviesEmbedding/embeddings/Regularizer/Square?
,MoviesEmbedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,MoviesEmbedding/embeddings/Regularizer/Const?
*MoviesEmbedding/embeddings/Regularizer/SumSum1MoviesEmbedding/embeddings/Regularizer/Square:y:05MoviesEmbedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*MoviesEmbedding/embeddings/Regularizer/Sum?
,MoviesEmbedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52.
,MoviesEmbedding/embeddings/Regularizer/mul/x?
*MoviesEmbedding/embeddings/Regularizer/mulMul5MoviesEmbedding/embeddings/Regularizer/mul/x:output:03MoviesEmbedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*MoviesEmbedding/embeddings/Regularizer/mul?
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp'usersembedding_embedding_lookup_3050719* 
_output_shapes
:
??*
dtype02=
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp?
,UsersEmbedding/embeddings/Regularizer/SquareSquareCUsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2.
,UsersEmbedding/embeddings/Regularizer/Square?
+UsersEmbedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+UsersEmbedding/embeddings/Regularizer/Const?
)UsersEmbedding/embeddings/Regularizer/SumSum0UsersEmbedding/embeddings/Regularizer/Square:y:04UsersEmbedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)UsersEmbedding/embeddings/Regularizer/Sum?
+UsersEmbedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52-
+UsersEmbedding/embeddings/Regularizer/mul/x?
)UsersEmbedding/embeddings/Regularizer/mulMul4UsersEmbedding/embeddings/Regularizer/mul/x:output:02UsersEmbedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)UsersEmbedding/embeddings/Regularizer/mul?
IdentityIdentityDotProduct/Squeeze:output:0!^MoviesEmbedding/embedding_lookup=^MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp ^UsersEmbedding/embedding_lookup<^UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:?????????:?????????: : 2D
 MoviesEmbedding/embedding_lookup MoviesEmbedding/embedding_lookup2|
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp2B
UsersEmbedding/embedding_lookupUsersEmbedding/embedding_lookup2z
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?

q
G__inference_DotProduct_layer_call_and_return_conditional_losses_3050481

inputs
inputs_1
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim~

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*,
_output_shapes
:??????????2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims_1/dim?
ExpandDims_1
ExpandDimsinputs_1ExpandDims_1/dim:output:0*
T0*,
_output_shapes
:??????????2
ExpandDims_1?
MatMulBatchMatMulV2ExpandDims:output:0ExpandDims_1:output:0*
T0*+
_output_shapes
:?????????2
MatMulM
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:2
Shapew
SqueezeSqueezeMatMul:output:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
2	
Squeezed
IdentityIdentitySqueeze:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs:PL
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?#
?
 __inference__traced_save_3050954
file_prefix9
5savev2_moviesembedding_embeddings_read_readvariableop8
4savev2_usersembedding_embeddings_read_readvariableop'
#savev2_sgd_iter_read_readvariableop	(
$savev2_sgd_decay_read_readvariableop0
,savev2_sgd_learning_rate_read_readvariableop+
'savev2_sgd_momentum_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:05savev2_moviesembedding_embeddings_read_readvariableop4savev2_usersembedding_embeddings_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*C
_input_shapes2
0: :
?K?:
??: : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
?K?:&"
 
_output_shapes
:
??:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	
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
: 
?
?
__inference_loss_fn_0_3050883Y
Emoviesembedding_embeddings_regularizer_square_readvariableop_resource:
?K?
identity??<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp?
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpEmoviesembedding_embeddings_regularizer_square_readvariableop_resource* 
_output_shapes
:
?K?*
dtype02>
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp?
-MoviesEmbedding/embeddings/Regularizer/SquareSquareDMoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
?K?2/
-MoviesEmbedding/embeddings/Regularizer/Square?
,MoviesEmbedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,MoviesEmbedding/embeddings/Regularizer/Const?
*MoviesEmbedding/embeddings/Regularizer/SumSum1MoviesEmbedding/embeddings/Regularizer/Square:y:05MoviesEmbedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*MoviesEmbedding/embeddings/Regularizer/Sum?
,MoviesEmbedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52.
,MoviesEmbedding/embeddings/Regularizer/mul/x?
*MoviesEmbedding/embeddings/Regularizer/mulMul5MoviesEmbedding/embeddings/Regularizer/mul/x:output:03MoviesEmbedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*MoviesEmbedding/embeddings/Regularizer/mul?
IdentityIdentity.MoviesEmbedding/embeddings/Regularizer/mul:z:0=^MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2|
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp
?
?
@__inference_MatrixFactorizationReccomender_layer_call_fn_3050594

usersinput
moviesinput
unknown:
??
	unknown_0:
?K?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
usersinputmoviesinputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *d
f_R]
[__inference_MatrixFactorizationReccomender_layer_call_and_return_conditional_losses_30505772
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:?????????:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:?????????
$
_user_specified_name
UsersInput:TP
'
_output_shapes
:?????????
%
_user_specified_nameMoviesInput
?
?
@__inference_MatrixFactorizationReccomender_layer_call_fn_3050503

usersinput
moviesinput
unknown:
??
	unknown_0:
?K?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
usersinputmoviesinputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *d
f_R]
[__inference_MatrixFactorizationReccomender_layer_call_and_return_conditional_losses_30504962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:?????????:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:?????????
$
_user_specified_name
UsersInput:TP
'
_output_shapes
:?????????
%
_user_specified_nameMoviesInput
?
e
I__inference_UsersFlatten_layer_call_and_return_conditional_losses_3050849

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????,  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
f
J__inference_MoviesFlatten_layer_call_and_return_conditional_losses_3050838

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????,  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
e
I__inference_UsersFlatten_layer_call_and_return_conditional_losses_3050467

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????,  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?/
?
[__inference_MatrixFactorizationReccomender_layer_call_and_return_conditional_losses_3050646

usersinput
moviesinput*
usersembedding_3050624:
??+
moviesembedding_3050627:
?K?
identity??'MoviesEmbedding/StatefulPartitionedCall?<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp?&UsersEmbedding/StatefulPartitionedCall?;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp?
&UsersEmbedding/StatefulPartitionedCallStatefulPartitionedCall
usersinputusersembedding_3050624*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_UsersEmbedding_layer_call_and_return_conditional_losses_30504292(
&UsersEmbedding/StatefulPartitionedCall?
'MoviesEmbedding/StatefulPartitionedCallStatefulPartitionedCallmoviesinputmoviesembedding_3050627*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_MoviesEmbedding_layer_call_and_return_conditional_losses_30504492)
'MoviesEmbedding/StatefulPartitionedCall?
MoviesFlatten/PartitionedCallPartitionedCall0MoviesEmbedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_MoviesFlatten_layer_call_and_return_conditional_losses_30504592
MoviesFlatten/PartitionedCall?
UsersFlatten/PartitionedCallPartitionedCall/UsersEmbedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_UsersFlatten_layer_call_and_return_conditional_losses_30504672
UsersFlatten/PartitionedCall?
DotProduct/PartitionedCallPartitionedCall&MoviesFlatten/PartitionedCall:output:0%UsersFlatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_DotProduct_layer_call_and_return_conditional_losses_30504812
DotProduct/PartitionedCall?
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpmoviesembedding_3050627* 
_output_shapes
:
?K?*
dtype02>
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp?
-MoviesEmbedding/embeddings/Regularizer/SquareSquareDMoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
?K?2/
-MoviesEmbedding/embeddings/Regularizer/Square?
,MoviesEmbedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,MoviesEmbedding/embeddings/Regularizer/Const?
*MoviesEmbedding/embeddings/Regularizer/SumSum1MoviesEmbedding/embeddings/Regularizer/Square:y:05MoviesEmbedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*MoviesEmbedding/embeddings/Regularizer/Sum?
,MoviesEmbedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52.
,MoviesEmbedding/embeddings/Regularizer/mul/x?
*MoviesEmbedding/embeddings/Regularizer/mulMul5MoviesEmbedding/embeddings/Regularizer/mul/x:output:03MoviesEmbedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*MoviesEmbedding/embeddings/Regularizer/mul?
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpusersembedding_3050624* 
_output_shapes
:
??*
dtype02=
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp?
,UsersEmbedding/embeddings/Regularizer/SquareSquareCUsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2.
,UsersEmbedding/embeddings/Regularizer/Square?
+UsersEmbedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+UsersEmbedding/embeddings/Regularizer/Const?
)UsersEmbedding/embeddings/Regularizer/SumSum0UsersEmbedding/embeddings/Regularizer/Square:y:04UsersEmbedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)UsersEmbedding/embeddings/Regularizer/Sum?
+UsersEmbedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52-
+UsersEmbedding/embeddings/Regularizer/mul/x?
)UsersEmbedding/embeddings/Regularizer/mulMul4UsersEmbedding/embeddings/Regularizer/mul/x:output:02UsersEmbedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)UsersEmbedding/embeddings/Regularizer/mul?
IdentityIdentity#DotProduct/PartitionedCall:output:0(^MoviesEmbedding/StatefulPartitionedCall=^MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp'^UsersEmbedding/StatefulPartitionedCall<^UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:?????????:?????????: : 2R
'MoviesEmbedding/StatefulPartitionedCall'MoviesEmbedding/StatefulPartitionedCall2|
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp2P
&UsersEmbedding/StatefulPartitionedCall&UsersEmbedding/StatefulPartitionedCall2z
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp:S O
'
_output_shapes
:?????????
$
_user_specified_name
UsersInput:TP
'
_output_shapes
:?????????
%
_user_specified_nameMoviesInput
?/
?
[__inference_MatrixFactorizationReccomender_layer_call_and_return_conditional_losses_3050496

inputs
inputs_1*
usersembedding_3050430:
??+
moviesembedding_3050450:
?K?
identity??'MoviesEmbedding/StatefulPartitionedCall?<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp?&UsersEmbedding/StatefulPartitionedCall?;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp?
&UsersEmbedding/StatefulPartitionedCallStatefulPartitionedCallinputsusersembedding_3050430*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_UsersEmbedding_layer_call_and_return_conditional_losses_30504292(
&UsersEmbedding/StatefulPartitionedCall?
'MoviesEmbedding/StatefulPartitionedCallStatefulPartitionedCallinputs_1moviesembedding_3050450*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_MoviesEmbedding_layer_call_and_return_conditional_losses_30504492)
'MoviesEmbedding/StatefulPartitionedCall?
MoviesFlatten/PartitionedCallPartitionedCall0MoviesEmbedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_MoviesFlatten_layer_call_and_return_conditional_losses_30504592
MoviesFlatten/PartitionedCall?
UsersFlatten/PartitionedCallPartitionedCall/UsersEmbedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_UsersFlatten_layer_call_and_return_conditional_losses_30504672
UsersFlatten/PartitionedCall?
DotProduct/PartitionedCallPartitionedCall&MoviesFlatten/PartitionedCall:output:0%UsersFlatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_DotProduct_layer_call_and_return_conditional_losses_30504812
DotProduct/PartitionedCall?
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpmoviesembedding_3050450* 
_output_shapes
:
?K?*
dtype02>
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp?
-MoviesEmbedding/embeddings/Regularizer/SquareSquareDMoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
?K?2/
-MoviesEmbedding/embeddings/Regularizer/Square?
,MoviesEmbedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,MoviesEmbedding/embeddings/Regularizer/Const?
*MoviesEmbedding/embeddings/Regularizer/SumSum1MoviesEmbedding/embeddings/Regularizer/Square:y:05MoviesEmbedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*MoviesEmbedding/embeddings/Regularizer/Sum?
,MoviesEmbedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52.
,MoviesEmbedding/embeddings/Regularizer/mul/x?
*MoviesEmbedding/embeddings/Regularizer/mulMul5MoviesEmbedding/embeddings/Regularizer/mul/x:output:03MoviesEmbedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*MoviesEmbedding/embeddings/Regularizer/mul?
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpusersembedding_3050430* 
_output_shapes
:
??*
dtype02=
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp?
,UsersEmbedding/embeddings/Regularizer/SquareSquareCUsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2.
,UsersEmbedding/embeddings/Regularizer/Square?
+UsersEmbedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+UsersEmbedding/embeddings/Regularizer/Const?
)UsersEmbedding/embeddings/Regularizer/SumSum0UsersEmbedding/embeddings/Regularizer/Square:y:04UsersEmbedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)UsersEmbedding/embeddings/Regularizer/Sum?
+UsersEmbedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52-
+UsersEmbedding/embeddings/Regularizer/mul/x?
)UsersEmbedding/embeddings/Regularizer/mulMul4UsersEmbedding/embeddings/Regularizer/mul/x:output:02UsersEmbedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)UsersEmbedding/embeddings/Regularizer/mul?
IdentityIdentity#DotProduct/PartitionedCall:output:0(^MoviesEmbedding/StatefulPartitionedCall=^MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp'^UsersEmbedding/StatefulPartitionedCall<^UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:?????????:?????????: : 2R
'MoviesEmbedding/StatefulPartitionedCall'MoviesEmbedding/StatefulPartitionedCall2|
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp2P
&UsersEmbedding/StatefulPartitionedCall&UsersEmbedding/StatefulPartitionedCall2z
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
J
.__inference_UsersFlatten_layer_call_fn_3050854

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_UsersFlatten_layer_call_and_return_conditional_losses_30504672
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
K__inference_UsersEmbedding_layer_call_and_return_conditional_losses_3050429

inputs,
embedding_lookup_3050417:
??
identity??;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp?embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_3050417Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*+
_class!
loc:@embedding_lookup/3050417*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@embedding_lookup/3050417*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_lookup_3050417* 
_output_shapes
:
??*
dtype02=
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp?
,UsersEmbedding/embeddings/Regularizer/SquareSquareCUsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2.
,UsersEmbedding/embeddings/Regularizer/Square?
+UsersEmbedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+UsersEmbedding/embeddings/Regularizer/Const?
)UsersEmbedding/embeddings/Regularizer/SumSum0UsersEmbedding/embeddings/Regularizer/Square:y:04UsersEmbedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)UsersEmbedding/embeddings/Regularizer/Sum?
+UsersEmbedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52-
+UsersEmbedding/embeddings/Regularizer/mul/x?
)UsersEmbedding/embeddings/Regularizer/mulMul4UsersEmbedding/embeddings/Regularizer/mul/x:output:02UsersEmbedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)UsersEmbedding/embeddings/Regularizer/mul?
IdentityIdentity$embedding_lookup/Identity_1:output:0<^UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2z
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?5
?
#__inference__traced_restore_3051000
file_prefix?
+assignvariableop_moviesembedding_embeddings:
?K?@
,assignvariableop_1_usersembedding_embeddings:
??%
assignvariableop_2_sgd_iter:	 &
assignvariableop_3_sgd_decay: .
$assignvariableop_4_sgd_learning_rate: )
assignvariableop_5_sgd_momentum: "
assignvariableop_6_total: "
assignvariableop_7_count: $
assignvariableop_8_total_1: $
assignvariableop_9_count_1: %
assignvariableop_10_total_2: %
assignvariableop_11_count_2: 
identity_13??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*H
_output_shapes6
4:::::::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp+assignvariableop_moviesembedding_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp,assignvariableop_1_usersembedding_embeddingsIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_sgd_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_sgd_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp$assignvariableop_4_sgd_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_sgd_momentumIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_totalIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_countIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOpassignvariableop_8_total_1Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_count_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_total_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_count_2Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_12Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_12?
Identity_13IdentityIdentity_12:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_13"#
identity_13Identity_13:output:0*-
_input_shapes
: : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
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
?/
?
[__inference_MatrixFactorizationReccomender_layer_call_and_return_conditional_losses_3050577

inputs
inputs_1*
usersembedding_3050555:
??+
moviesembedding_3050558:
?K?
identity??'MoviesEmbedding/StatefulPartitionedCall?<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp?&UsersEmbedding/StatefulPartitionedCall?;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp?
&UsersEmbedding/StatefulPartitionedCallStatefulPartitionedCallinputsusersembedding_3050555*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_UsersEmbedding_layer_call_and_return_conditional_losses_30504292(
&UsersEmbedding/StatefulPartitionedCall?
'MoviesEmbedding/StatefulPartitionedCallStatefulPartitionedCallinputs_1moviesembedding_3050558*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_MoviesEmbedding_layer_call_and_return_conditional_losses_30504492)
'MoviesEmbedding/StatefulPartitionedCall?
MoviesFlatten/PartitionedCallPartitionedCall0MoviesEmbedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_MoviesFlatten_layer_call_and_return_conditional_losses_30504592
MoviesFlatten/PartitionedCall?
UsersFlatten/PartitionedCallPartitionedCall/UsersEmbedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_UsersFlatten_layer_call_and_return_conditional_losses_30504672
UsersFlatten/PartitionedCall?
DotProduct/PartitionedCallPartitionedCall&MoviesFlatten/PartitionedCall:output:0%UsersFlatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_DotProduct_layer_call_and_return_conditional_losses_30504812
DotProduct/PartitionedCall?
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpmoviesembedding_3050558* 
_output_shapes
:
?K?*
dtype02>
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp?
-MoviesEmbedding/embeddings/Regularizer/SquareSquareDMoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
?K?2/
-MoviesEmbedding/embeddings/Regularizer/Square?
,MoviesEmbedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,MoviesEmbedding/embeddings/Regularizer/Const?
*MoviesEmbedding/embeddings/Regularizer/SumSum1MoviesEmbedding/embeddings/Regularizer/Square:y:05MoviesEmbedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*MoviesEmbedding/embeddings/Regularizer/Sum?
,MoviesEmbedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52.
,MoviesEmbedding/embeddings/Regularizer/mul/x?
*MoviesEmbedding/embeddings/Regularizer/mulMul5MoviesEmbedding/embeddings/Regularizer/mul/x:output:03MoviesEmbedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*MoviesEmbedding/embeddings/Regularizer/mul?
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpusersembedding_3050555* 
_output_shapes
:
??*
dtype02=
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp?
,UsersEmbedding/embeddings/Regularizer/SquareSquareCUsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2.
,UsersEmbedding/embeddings/Regularizer/Square?
+UsersEmbedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+UsersEmbedding/embeddings/Regularizer/Const?
)UsersEmbedding/embeddings/Regularizer/SumSum0UsersEmbedding/embeddings/Regularizer/Square:y:04UsersEmbedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)UsersEmbedding/embeddings/Regularizer/Sum?
+UsersEmbedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52-
+UsersEmbedding/embeddings/Regularizer/mul/x?
)UsersEmbedding/embeddings/Regularizer/mulMul4UsersEmbedding/embeddings/Regularizer/mul/x:output:02UsersEmbedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)UsersEmbedding/embeddings/Regularizer/mul?
IdentityIdentity#DotProduct/PartitionedCall:output:0(^MoviesEmbedding/StatefulPartitionedCall=^MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp'^UsersEmbedding/StatefulPartitionedCall<^UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:?????????:?????????: : 2R
'MoviesEmbedding/StatefulPartitionedCall'MoviesEmbedding/StatefulPartitionedCall2|
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp2P
&UsersEmbedding/StatefulPartitionedCall&UsersEmbedding/StatefulPartitionedCall2z
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
X
,__inference_DotProduct_layer_call_fn_3050872
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_DotProduct_layer_call_and_return_conditional_losses_30504812
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????:??????????:R N
(
_output_shapes
:??????????
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/1
?
?
@__inference_MatrixFactorizationReccomender_layer_call_fn_3050774
inputs_0
inputs_1
unknown:
??
	unknown_0:
?K?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *d
f_R]
[__inference_MatrixFactorizationReccomender_layer_call_and_return_conditional_losses_30505772
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:?????????:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
?
%__inference_signature_wrapper_3050674
moviesinput

usersinput
unknown:
??
	unknown_0:
?K?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
usersinputmoviesinputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__wrapped_model_30504042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:?????????:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:?????????
%
_user_specified_nameMoviesInput:SO
'
_output_shapes
:?????????
$
_user_specified_name
UsersInput
?
f
J__inference_MoviesFlatten_layer_call_and_return_conditional_losses_3050459

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????,  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
K
/__inference_MoviesFlatten_layer_call_fn_3050843

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_MoviesFlatten_layer_call_and_return_conditional_losses_30504592
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????:T P
,
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
1__inference_MoviesEmbedding_layer_call_fn_3050803

inputs
unknown:
?K?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_MoviesEmbedding_layer_call_and_return_conditional_losses_30504492
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
K__inference_UsersEmbedding_layer_call_and_return_conditional_losses_3050825

inputs,
embedding_lookup_3050813:
??
identity??;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp?embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_3050813Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*+
_class!
loc:@embedding_lookup/3050813*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@embedding_lookup/3050813*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_lookup_3050813* 
_output_shapes
:
??*
dtype02=
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp?
,UsersEmbedding/embeddings/Regularizer/SquareSquareCUsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2.
,UsersEmbedding/embeddings/Regularizer/Square?
+UsersEmbedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+UsersEmbedding/embeddings/Regularizer/Const?
)UsersEmbedding/embeddings/Regularizer/SumSum0UsersEmbedding/embeddings/Regularizer/Square:y:04UsersEmbedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)UsersEmbedding/embeddings/Regularizer/Sum?
+UsersEmbedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52-
+UsersEmbedding/embeddings/Regularizer/mul/x?
)UsersEmbedding/embeddings/Regularizer/mulMul4UsersEmbedding/embeddings/Regularizer/mul/x:output:02UsersEmbedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)UsersEmbedding/embeddings/Regularizer/mul?
IdentityIdentity$embedding_lookup/Identity_1:output:0<^UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2z
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
0__inference_UsersEmbedding_layer_call_fn_3050832

inputs
unknown:
??
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_UsersEmbedding_layer_call_and_return_conditional_losses_30504292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

s
G__inference_DotProduct_layer_call_and_return_conditional_losses_3050866
inputs_0
inputs_1
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim?

ExpandDims
ExpandDimsinputs_0ExpandDims/dim:output:0*
T0*,
_output_shapes
:??????????2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims_1/dim?
ExpandDims_1
ExpandDimsinputs_1ExpandDims_1/dim:output:0*
T0*,
_output_shapes
:??????????2
ExpandDims_1?
MatMulBatchMatMulV2ExpandDims:output:0ExpandDims_1:output:0*
T0*+
_output_shapes
:?????????2
MatMulM
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:2
Shapew
SqueezeSqueezeMatMul:output:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
2	
Squeezed
IdentityIdentitySqueeze:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:??????????:??????????:R N
(
_output_shapes
:??????????
"
_user_specified_name
inputs/0:RN
(
_output_shapes
:??????????
"
_user_specified_name
inputs/1
?5
?
"__inference__wrapped_model_3050404

usersinput
moviesinputZ
Fmatrixfactorizationreccomender_usersembedding_embedding_lookup_3050381:
??[
Gmatrixfactorizationreccomender_moviesembedding_embedding_lookup_3050387:
?K?
identity???MatrixFactorizationReccomender/MoviesEmbedding/embedding_lookup?>MatrixFactorizationReccomender/UsersEmbedding/embedding_lookup?
2MatrixFactorizationReccomender/UsersEmbedding/CastCast
usersinput*

DstT0*

SrcT0*'
_output_shapes
:?????????24
2MatrixFactorizationReccomender/UsersEmbedding/Cast?
>MatrixFactorizationReccomender/UsersEmbedding/embedding_lookupResourceGatherFmatrixfactorizationreccomender_usersembedding_embedding_lookup_30503816MatrixFactorizationReccomender/UsersEmbedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*Y
_classO
MKloc:@MatrixFactorizationReccomender/UsersEmbedding/embedding_lookup/3050381*,
_output_shapes
:??????????*
dtype02@
>MatrixFactorizationReccomender/UsersEmbedding/embedding_lookup?
GMatrixFactorizationReccomender/UsersEmbedding/embedding_lookup/IdentityIdentityGMatrixFactorizationReccomender/UsersEmbedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*Y
_classO
MKloc:@MatrixFactorizationReccomender/UsersEmbedding/embedding_lookup/3050381*,
_output_shapes
:??????????2I
GMatrixFactorizationReccomender/UsersEmbedding/embedding_lookup/Identity?
IMatrixFactorizationReccomender/UsersEmbedding/embedding_lookup/Identity_1IdentityPMatrixFactorizationReccomender/UsersEmbedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2K
IMatrixFactorizationReccomender/UsersEmbedding/embedding_lookup/Identity_1?
3MatrixFactorizationReccomender/MoviesEmbedding/CastCastmoviesinput*

DstT0*

SrcT0*'
_output_shapes
:?????????25
3MatrixFactorizationReccomender/MoviesEmbedding/Cast?
?MatrixFactorizationReccomender/MoviesEmbedding/embedding_lookupResourceGatherGmatrixfactorizationreccomender_moviesembedding_embedding_lookup_30503877MatrixFactorizationReccomender/MoviesEmbedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*Z
_classP
NLloc:@MatrixFactorizationReccomender/MoviesEmbedding/embedding_lookup/3050387*,
_output_shapes
:??????????*
dtype02A
?MatrixFactorizationReccomender/MoviesEmbedding/embedding_lookup?
HMatrixFactorizationReccomender/MoviesEmbedding/embedding_lookup/IdentityIdentityHMatrixFactorizationReccomender/MoviesEmbedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*Z
_classP
NLloc:@MatrixFactorizationReccomender/MoviesEmbedding/embedding_lookup/3050387*,
_output_shapes
:??????????2J
HMatrixFactorizationReccomender/MoviesEmbedding/embedding_lookup/Identity?
JMatrixFactorizationReccomender/MoviesEmbedding/embedding_lookup/Identity_1IdentityQMatrixFactorizationReccomender/MoviesEmbedding/embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2L
JMatrixFactorizationReccomender/MoviesEmbedding/embedding_lookup/Identity_1?
2MatrixFactorizationReccomender/MoviesFlatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????,  24
2MatrixFactorizationReccomender/MoviesFlatten/Const?
4MatrixFactorizationReccomender/MoviesFlatten/ReshapeReshapeSMatrixFactorizationReccomender/MoviesEmbedding/embedding_lookup/Identity_1:output:0;MatrixFactorizationReccomender/MoviesFlatten/Const:output:0*
T0*(
_output_shapes
:??????????26
4MatrixFactorizationReccomender/MoviesFlatten/Reshape?
1MatrixFactorizationReccomender/UsersFlatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????,  23
1MatrixFactorizationReccomender/UsersFlatten/Const?
3MatrixFactorizationReccomender/UsersFlatten/ReshapeReshapeRMatrixFactorizationReccomender/UsersEmbedding/embedding_lookup/Identity_1:output:0:MatrixFactorizationReccomender/UsersFlatten/Const:output:0*
T0*(
_output_shapes
:??????????25
3MatrixFactorizationReccomender/UsersFlatten/Reshape?
8MatrixFactorizationReccomender/DotProduct/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2:
8MatrixFactorizationReccomender/DotProduct/ExpandDims/dim?
4MatrixFactorizationReccomender/DotProduct/ExpandDims
ExpandDims=MatrixFactorizationReccomender/MoviesFlatten/Reshape:output:0AMatrixFactorizationReccomender/DotProduct/ExpandDims/dim:output:0*
T0*,
_output_shapes
:??????????26
4MatrixFactorizationReccomender/DotProduct/ExpandDims?
:MatrixFactorizationReccomender/DotProduct/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2<
:MatrixFactorizationReccomender/DotProduct/ExpandDims_1/dim?
6MatrixFactorizationReccomender/DotProduct/ExpandDims_1
ExpandDims<MatrixFactorizationReccomender/UsersFlatten/Reshape:output:0CMatrixFactorizationReccomender/DotProduct/ExpandDims_1/dim:output:0*
T0*,
_output_shapes
:??????????28
6MatrixFactorizationReccomender/DotProduct/ExpandDims_1?
0MatrixFactorizationReccomender/DotProduct/MatMulBatchMatMulV2=MatrixFactorizationReccomender/DotProduct/ExpandDims:output:0?MatrixFactorizationReccomender/DotProduct/ExpandDims_1:output:0*
T0*+
_output_shapes
:?????????22
0MatrixFactorizationReccomender/DotProduct/MatMul?
/MatrixFactorizationReccomender/DotProduct/ShapeShape9MatrixFactorizationReccomender/DotProduct/MatMul:output:0*
T0*
_output_shapes
:21
/MatrixFactorizationReccomender/DotProduct/Shape?
1MatrixFactorizationReccomender/DotProduct/SqueezeSqueeze9MatrixFactorizationReccomender/DotProduct/MatMul:output:0*
T0*'
_output_shapes
:?????????*
squeeze_dims
23
1MatrixFactorizationReccomender/DotProduct/Squeeze?
IdentityIdentity:MatrixFactorizationReccomender/DotProduct/Squeeze:output:0@^MatrixFactorizationReccomender/MoviesEmbedding/embedding_lookup?^MatrixFactorizationReccomender/UsersEmbedding/embedding_lookup*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:?????????:?????????: : 2?
?MatrixFactorizationReccomender/MoviesEmbedding/embedding_lookup?MatrixFactorizationReccomender/MoviesEmbedding/embedding_lookup2?
>MatrixFactorizationReccomender/UsersEmbedding/embedding_lookup>MatrixFactorizationReccomender/UsersEmbedding/embedding_lookup:S O
'
_output_shapes
:?????????
$
_user_specified_name
UsersInput:TP
'
_output_shapes
:?????????
%
_user_specified_nameMoviesInput
?
?
L__inference_MoviesEmbedding_layer_call_and_return_conditional_losses_3050796

inputs,
embedding_lookup_3050784:
?K?
identity??<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp?embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_3050784Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*+
_class!
loc:@embedding_lookup/3050784*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@embedding_lookup/3050784*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_lookup_3050784* 
_output_shapes
:
?K?*
dtype02>
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp?
-MoviesEmbedding/embeddings/Regularizer/SquareSquareDMoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
?K?2/
-MoviesEmbedding/embeddings/Regularizer/Square?
,MoviesEmbedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,MoviesEmbedding/embeddings/Regularizer/Const?
*MoviesEmbedding/embeddings/Regularizer/SumSum1MoviesEmbedding/embeddings/Regularizer/Square:y:05MoviesEmbedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*MoviesEmbedding/embeddings/Regularizer/Sum?
,MoviesEmbedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52.
,MoviesEmbedding/embeddings/Regularizer/mul/x?
*MoviesEmbedding/embeddings/Regularizer/mulMul5MoviesEmbedding/embeddings/Regularizer/mul/x:output:03MoviesEmbedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*MoviesEmbedding/embeddings/Regularizer/mul?
IdentityIdentity$embedding_lookup/Identity_1:output:0=^MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2|
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
L__inference_MoviesEmbedding_layer_call_and_return_conditional_losses_3050449

inputs,
embedding_lookup_3050437:
?K?
identity??<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp?embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_3050437Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*+
_class!
loc:@embedding_lookup/3050437*,
_output_shapes
:??????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@embedding_lookup/3050437*,
_output_shapes
:??????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*,
_output_shapes
:??????????2
embedding_lookup/Identity_1?
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_lookup_3050437* 
_output_shapes
:
?K?*
dtype02>
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp?
-MoviesEmbedding/embeddings/Regularizer/SquareSquareDMoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
?K?2/
-MoviesEmbedding/embeddings/Regularizer/Square?
,MoviesEmbedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,MoviesEmbedding/embeddings/Regularizer/Const?
*MoviesEmbedding/embeddings/Regularizer/SumSum1MoviesEmbedding/embeddings/Regularizer/Square:y:05MoviesEmbedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*MoviesEmbedding/embeddings/Regularizer/Sum?
,MoviesEmbedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52.
,MoviesEmbedding/embeddings/Regularizer/mul/x?
*MoviesEmbedding/embeddings/Regularizer/mulMul5MoviesEmbedding/embeddings/Regularizer/mul/x:output:03MoviesEmbedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*MoviesEmbedding/embeddings/Regularizer/mul?
IdentityIdentity$embedding_lookup/Identity_1:output:0=^MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp^embedding_lookup*
T0*,
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2|
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?/
?
[__inference_MatrixFactorizationReccomender_layer_call_and_return_conditional_losses_3050620

usersinput
moviesinput*
usersembedding_3050598:
??+
moviesembedding_3050601:
?K?
identity??'MoviesEmbedding/StatefulPartitionedCall?<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp?&UsersEmbedding/StatefulPartitionedCall?;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp?
&UsersEmbedding/StatefulPartitionedCallStatefulPartitionedCall
usersinputusersembedding_3050598*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_UsersEmbedding_layer_call_and_return_conditional_losses_30504292(
&UsersEmbedding/StatefulPartitionedCall?
'MoviesEmbedding/StatefulPartitionedCallStatefulPartitionedCallmoviesinputmoviesembedding_3050601*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_MoviesEmbedding_layer_call_and_return_conditional_losses_30504492)
'MoviesEmbedding/StatefulPartitionedCall?
MoviesFlatten/PartitionedCallPartitionedCall0MoviesEmbedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_MoviesFlatten_layer_call_and_return_conditional_losses_30504592
MoviesFlatten/PartitionedCall?
UsersFlatten/PartitionedCallPartitionedCall/UsersEmbedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_UsersFlatten_layer_call_and_return_conditional_losses_30504672
UsersFlatten/PartitionedCall?
DotProduct/PartitionedCallPartitionedCall&MoviesFlatten/PartitionedCall:output:0%UsersFlatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_DotProduct_layer_call_and_return_conditional_losses_30504812
DotProduct/PartitionedCall?
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpmoviesembedding_3050601* 
_output_shapes
:
?K?*
dtype02>
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp?
-MoviesEmbedding/embeddings/Regularizer/SquareSquareDMoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
?K?2/
-MoviesEmbedding/embeddings/Regularizer/Square?
,MoviesEmbedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,MoviesEmbedding/embeddings/Regularizer/Const?
*MoviesEmbedding/embeddings/Regularizer/SumSum1MoviesEmbedding/embeddings/Regularizer/Square:y:05MoviesEmbedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*MoviesEmbedding/embeddings/Regularizer/Sum?
,MoviesEmbedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52.
,MoviesEmbedding/embeddings/Regularizer/mul/x?
*MoviesEmbedding/embeddings/Regularizer/mulMul5MoviesEmbedding/embeddings/Regularizer/mul/x:output:03MoviesEmbedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*MoviesEmbedding/embeddings/Regularizer/mul?
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpusersembedding_3050598* 
_output_shapes
:
??*
dtype02=
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp?
,UsersEmbedding/embeddings/Regularizer/SquareSquareCUsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2.
,UsersEmbedding/embeddings/Regularizer/Square?
+UsersEmbedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+UsersEmbedding/embeddings/Regularizer/Const?
)UsersEmbedding/embeddings/Regularizer/SumSum0UsersEmbedding/embeddings/Regularizer/Square:y:04UsersEmbedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)UsersEmbedding/embeddings/Regularizer/Sum?
+UsersEmbedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52-
+UsersEmbedding/embeddings/Regularizer/mul/x?
)UsersEmbedding/embeddings/Regularizer/mulMul4UsersEmbedding/embeddings/Regularizer/mul/x:output:02UsersEmbedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)UsersEmbedding/embeddings/Regularizer/mul?
IdentityIdentity#DotProduct/PartitionedCall:output:0(^MoviesEmbedding/StatefulPartitionedCall=^MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp'^UsersEmbedding/StatefulPartitionedCall<^UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:?????????:?????????: : 2R
'MoviesEmbedding/StatefulPartitionedCall'MoviesEmbedding/StatefulPartitionedCall2|
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp2P
&UsersEmbedding/StatefulPartitionedCall&UsersEmbedding/StatefulPartitionedCall2z
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp:S O
'
_output_shapes
:?????????
$
_user_specified_name
UsersInput:TP
'
_output_shapes
:?????????
%
_user_specified_nameMoviesInput
?
?
__inference_loss_fn_1_3050894X
Dusersembedding_embeddings_regularizer_square_readvariableop_resource:
??
identity??;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp?
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpDusersembedding_embeddings_regularizer_square_readvariableop_resource* 
_output_shapes
:
??*
dtype02=
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp?
,UsersEmbedding/embeddings/Regularizer/SquareSquareCUsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0* 
_output_shapes
:
??2.
,UsersEmbedding/embeddings/Regularizer/Square?
+UsersEmbedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+UsersEmbedding/embeddings/Regularizer/Const?
)UsersEmbedding/embeddings/Regularizer/SumSum0UsersEmbedding/embeddings/Regularizer/Square:y:04UsersEmbedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)UsersEmbedding/embeddings/Regularizer/Sum?
+UsersEmbedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?7?52-
+UsersEmbedding/embeddings/Regularizer/mul/x?
)UsersEmbedding/embeddings/Regularizer/mulMul4UsersEmbedding/embeddings/Regularizer/mul/x:output:02UsersEmbedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)UsersEmbedding/embeddings/Regularizer/mul?
IdentityIdentity-UsersEmbedding/embeddings/Regularizer/mul:z:0<^UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2z
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp
?
?
@__inference_MatrixFactorizationReccomender_layer_call_fn_3050764
inputs_0
inputs_1
unknown:
??
	unknown_0:
?K?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *d
f_R]
[__inference_MatrixFactorizationReccomender_layer_call_and_return_conditional_losses_30504962
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:?????????:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
MoviesInput4
serving_default_MoviesInput:0?????????
A

UsersInput3
serving_default_UsersInput:0?????????>

DotProduct0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?:
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
	optimizer
	regularization_losses

trainable_variables
	variables
	keras_api

signatures
*Y&call_and_return_all_conditional_losses
Z__call__
[_default_save_signature"?7
_tf_keras_network?7{"name": "MatrixFactorizationReccomender", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "MatrixFactorizationReccomender", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "MoviesInput"}, "name": "MoviesInput", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "UsersInput"}, "name": "UsersInput", "inbound_nodes": []}, {"class_name": "Embedding", "config": {"name": "MoviesEmbedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 9724, "output_dim": 300, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": {"class_name": "L2", "config": {"l2": 9.999999974752427e-07}}, "activity_regularizer": null, "embeddings_constraint": {"class_name": "NonNeg", "config": {}}, "mask_zero": false, "input_length": null}, "name": "MoviesEmbedding", "inbound_nodes": [[["MoviesInput", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "UsersEmbedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 610, "output_dim": 300, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": {"class_name": "L2", "config": {"l2": 9.999999974752427e-07}}, "activity_regularizer": null, "embeddings_constraint": {"class_name": "NonNeg", "config": {}}, "mask_zero": false, "input_length": null}, "name": "UsersEmbedding", "inbound_nodes": [[["UsersInput", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "MoviesFlatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "MoviesFlatten", "inbound_nodes": [[["MoviesEmbedding", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "UsersFlatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "UsersFlatten", "inbound_nodes": [[["UsersEmbedding", 0, 0, {}]]]}, {"class_name": "Dot", "config": {"name": "DotProduct", "trainable": true, "dtype": "float32", "axes": 1, "normalize": false}, "name": "DotProduct", "inbound_nodes": [[["MoviesFlatten", 0, 0, {}], ["UsersFlatten", 0, 0, {}]]]}], "input_layers": [["UsersInput", 0, 0], ["MoviesInput", 0, 0]], "output_layers": [["DotProduct", 0, 0]]}, "shared_object_id": 13, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}], "is_graph_network": true, "save_spec": [{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "UsersInput"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "MoviesInput"]}], "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "MatrixFactorizationReccomender", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "MoviesInput"}, "name": "MoviesInput", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "UsersInput"}, "name": "UsersInput", "inbound_nodes": [], "shared_object_id": 1}, {"class_name": "Embedding", "config": {"name": "MoviesEmbedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 9724, "output_dim": 300, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 2}, "embeddings_regularizer": {"class_name": "L2", "config": {"l2": 9.999999974752427e-07}, "shared_object_id": 3}, "activity_regularizer": null, "embeddings_constraint": {"class_name": "NonNeg", "config": {}, "shared_object_id": 4}, "mask_zero": false, "input_length": null}, "name": "MoviesEmbedding", "inbound_nodes": [[["MoviesInput", 0, 0, {}]]], "shared_object_id": 5}, {"class_name": "Embedding", "config": {"name": "UsersEmbedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 610, "output_dim": 300, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 6}, "embeddings_regularizer": {"class_name": "L2", "config": {"l2": 9.999999974752427e-07}, "shared_object_id": 7}, "activity_regularizer": null, "embeddings_constraint": {"class_name": "NonNeg", "config": {}, "shared_object_id": 8}, "mask_zero": false, "input_length": null}, "name": "UsersEmbedding", "inbound_nodes": [[["UsersInput", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "Flatten", "config": {"name": "MoviesFlatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "MoviesFlatten", "inbound_nodes": [[["MoviesEmbedding", 0, 0, {}]]], "shared_object_id": 10}, {"class_name": "Flatten", "config": {"name": "UsersFlatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "UsersFlatten", "inbound_nodes": [[["UsersEmbedding", 0, 0, {}]]], "shared_object_id": 11}, {"class_name": "Dot", "config": {"name": "DotProduct", "trainable": true, "dtype": "float32", "axes": 1, "normalize": false}, "name": "DotProduct", "inbound_nodes": [[["MoviesFlatten", 0, 0, {}], ["UsersFlatten", 0, 0, {}]]], "shared_object_id": 12}], "input_layers": [["UsersInput", 0, 0], ["MoviesInput", 0, 0]], "output_layers": [["DotProduct", 0, 0]]}}, "training_config": {"loss": "mean_squared_error", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}, "shared_object_id": 16}, {"class_name": "MeanMetricWrapper", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}, "shared_object_id": 17}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": 0.009999999776482582, "decay": 0.0, "momentum": 0.0, "nesterov": false}}}}
?
_init_input_shape"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "MoviesInput", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "MoviesInput"}}
?
_init_input_shape"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "UsersInput", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "UsersInput"}}
?	

embeddings
regularization_losses
trainable_variables
	variables
	keras_api
*\&call_and_return_all_conditional_losses
]__call__"?
_tf_keras_layer?{"name": "MoviesEmbedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "class_name": "Embedding", "config": {"name": "MoviesEmbedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 9724, "output_dim": 300, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 2}, "embeddings_regularizer": {"class_name": "L2", "config": {"l2": 9.999999974752427e-07}, "shared_object_id": 3}, "activity_regularizer": null, "embeddings_constraint": {"class_name": "NonNeg", "config": {}, "shared_object_id": 4}, "mask_zero": false, "input_length": null}, "inbound_nodes": [[["MoviesInput", 0, 0, {}]]], "shared_object_id": 5, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
?	

embeddings
regularization_losses
trainable_variables
	variables
	keras_api
*^&call_and_return_all_conditional_losses
___call__"?
_tf_keras_layer?{"name": "UsersEmbedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "class_name": "Embedding", "config": {"name": "UsersEmbedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 610, "output_dim": 300, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 6}, "embeddings_regularizer": {"class_name": "L2", "config": {"l2": 9.999999974752427e-07}, "shared_object_id": 7}, "activity_regularizer": null, "embeddings_constraint": {"class_name": "NonNeg", "config": {}, "shared_object_id": 8}, "mask_zero": false, "input_length": null}, "inbound_nodes": [[["UsersInput", 0, 0, {}]]], "shared_object_id": 9, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
?
regularization_losses
trainable_variables
	variables
	keras_api
*`&call_and_return_all_conditional_losses
a__call__"?
_tf_keras_layer?{"name": "MoviesFlatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "MoviesFlatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["MoviesEmbedding", 0, 0, {}]]], "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 18}}
?
regularization_losses
trainable_variables
 	variables
!	keras_api
*b&call_and_return_all_conditional_losses
c__call__"?
_tf_keras_layer?{"name": "UsersFlatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "UsersFlatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["UsersEmbedding", 0, 0, {}]]], "shared_object_id": 11, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 19}}
?
"regularization_losses
#trainable_variables
$	variables
%	keras_api
*d&call_and_return_all_conditional_losses
e__call__"?
_tf_keras_layer?{"name": "DotProduct", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dot", "config": {"name": "DotProduct", "trainable": true, "dtype": "float32", "axes": 1, "normalize": false}, "inbound_nodes": [[["MoviesFlatten", 0, 0, {}], ["UsersFlatten", 0, 0, {}]]], "shared_object_id": 12, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 300]}, {"class_name": "TensorShape", "items": [null, 300]}]}
I
&iter
	'decay
(learning_rate
)momentum"
	optimizer
.
f0
g1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
	regularization_losses

trainable_variables
*layer_metrics

+layers
	variables
,layer_regularization_losses
-non_trainable_variables
.metrics
Z__call__
[_default_save_signature
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
,
hserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.:,
?K?2MoviesEmbedding/embeddings
'
f0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
?
regularization_losses
trainable_variables
/layer_metrics

0layers
	variables
1layer_regularization_losses
2non_trainable_variables
3metrics
]__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
-:+
??2UsersEmbedding/embeddings
'
g0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
?
regularization_losses
trainable_variables
4layer_metrics

5layers
	variables
6layer_regularization_losses
7non_trainable_variables
8metrics
___call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
regularization_losses
trainable_variables
9layer_metrics

:layers
	variables
;layer_regularization_losses
<non_trainable_variables
=metrics
a__call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
regularization_losses
trainable_variables
>layer_metrics

?layers
 	variables
@layer_regularization_losses
Anon_trainable_variables
Bmetrics
c__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
"regularization_losses
#trainable_variables
Clayer_metrics

Dlayers
$	variables
Elayer_regularization_losses
Fnon_trainable_variables
Gmetrics
e__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum
 "
trackable_dict_wrapper
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
H0
I1
J2"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
f0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
g0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	Ktotal
	Lcount
M	variables
N	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 20}
?
	Ototal
	Pcount
Q
_fn_kwargs
R	variables
S	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "mae", "dtype": "float32", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}, "shared_object_id": 16}
?
	Ttotal
	Ucount
V
_fn_kwargs
W	variables
X	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "mse", "dtype": "float32", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}, "shared_object_id": 17}
:  (2total
:  (2count
.
K0
L1"
trackable_list_wrapper
-
M	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
O0
P1"
trackable_list_wrapper
-
R	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
T0
U1"
trackable_list_wrapper
-
W	variables"
_generic_user_object
?2?
[__inference_MatrixFactorizationReccomender_layer_call_and_return_conditional_losses_3050714
[__inference_MatrixFactorizationReccomender_layer_call_and_return_conditional_losses_3050754
[__inference_MatrixFactorizationReccomender_layer_call_and_return_conditional_losses_3050620
[__inference_MatrixFactorizationReccomender_layer_call_and_return_conditional_losses_3050646?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
@__inference_MatrixFactorizationReccomender_layer_call_fn_3050503
@__inference_MatrixFactorizationReccomender_layer_call_fn_3050764
@__inference_MatrixFactorizationReccomender_layer_call_fn_3050774
@__inference_MatrixFactorizationReccomender_layer_call_fn_3050594?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
"__inference__wrapped_model_3050404?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *U?R
P?M
$?!

UsersInput?????????
%?"
MoviesInput?????????
?2?
L__inference_MoviesEmbedding_layer_call_and_return_conditional_losses_3050796?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_MoviesEmbedding_layer_call_fn_3050803?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_UsersEmbedding_layer_call_and_return_conditional_losses_3050825?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_UsersEmbedding_layer_call_fn_3050832?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_MoviesFlatten_layer_call_and_return_conditional_losses_3050838?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_MoviesFlatten_layer_call_fn_3050843?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
I__inference_UsersFlatten_layer_call_and_return_conditional_losses_3050849?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_UsersFlatten_layer_call_fn_3050854?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_DotProduct_layer_call_and_return_conditional_losses_3050866?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_DotProduct_layer_call_fn_3050872?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_loss_fn_0_3050883?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_1_3050894?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
%__inference_signature_wrapper_3050674MoviesInput
UsersInput"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
G__inference_DotProduct_layer_call_and_return_conditional_losses_3050866?\?Y
R?O
M?J
#? 
inputs/0??????????
#? 
inputs/1??????????
? "%?"
?
0?????????
? ?
,__inference_DotProduct_layer_call_fn_3050872x\?Y
R?O
M?J
#? 
inputs/0??????????
#? 
inputs/1??????????
? "???????????
[__inference_MatrixFactorizationReccomender_layer_call_and_return_conditional_losses_3050620?g?d
]?Z
P?M
$?!

UsersInput?????????
%?"
MoviesInput?????????
p 

 
? "%?"
?
0?????????
? ?
[__inference_MatrixFactorizationReccomender_layer_call_and_return_conditional_losses_3050646?g?d
]?Z
P?M
$?!

UsersInput?????????
%?"
MoviesInput?????????
p

 
? "%?"
?
0?????????
? ?
[__inference_MatrixFactorizationReccomender_layer_call_and_return_conditional_losses_3050714?b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p 

 
? "%?"
?
0?????????
? ?
[__inference_MatrixFactorizationReccomender_layer_call_and_return_conditional_losses_3050754?b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p

 
? "%?"
?
0?????????
? ?
@__inference_MatrixFactorizationReccomender_layer_call_fn_3050503?g?d
]?Z
P?M
$?!

UsersInput?????????
%?"
MoviesInput?????????
p 

 
? "???????????
@__inference_MatrixFactorizationReccomender_layer_call_fn_3050594?g?d
]?Z
P?M
$?!

UsersInput?????????
%?"
MoviesInput?????????
p

 
? "???????????
@__inference_MatrixFactorizationReccomender_layer_call_fn_3050764?b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p 

 
? "???????????
@__inference_MatrixFactorizationReccomender_layer_call_fn_3050774?b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????
p

 
? "???????????
L__inference_MoviesEmbedding_layer_call_and_return_conditional_losses_3050796`/?,
%?"
 ?
inputs?????????
? "*?'
 ?
0??????????
? ?
1__inference_MoviesEmbedding_layer_call_fn_3050803S/?,
%?"
 ?
inputs?????????
? "????????????
J__inference_MoviesFlatten_layer_call_and_return_conditional_losses_3050838^4?1
*?'
%?"
inputs??????????
? "&?#
?
0??????????
? ?
/__inference_MoviesFlatten_layer_call_fn_3050843Q4?1
*?'
%?"
inputs??????????
? "????????????
K__inference_UsersEmbedding_layer_call_and_return_conditional_losses_3050825`/?,
%?"
 ?
inputs?????????
? "*?'
 ?
0??????????
? ?
0__inference_UsersEmbedding_layer_call_fn_3050832S/?,
%?"
 ?
inputs?????????
? "????????????
I__inference_UsersFlatten_layer_call_and_return_conditional_losses_3050849^4?1
*?'
%?"
inputs??????????
? "&?#
?
0??????????
? ?
.__inference_UsersFlatten_layer_call_fn_3050854Q4?1
*?'
%?"
inputs??????????
? "????????????
"__inference__wrapped_model_3050404?_?\
U?R
P?M
$?!

UsersInput?????????
%?"
MoviesInput?????????
? "7?4
2

DotProduct$?!

DotProduct?????????<
__inference_loss_fn_0_3050883?

? 
? "? <
__inference_loss_fn_1_3050894?

? 
? "? ?
%__inference_signature_wrapper_3050674?w?t
? 
m?j
4
MoviesInput%?"
MoviesInput?????????
2

UsersInput$?!

UsersInput?????????"7?4
2

DotProduct$?!

DotProduct?????????