É
·
B
AssignVariableOp
resource
value"dtype"
dtypetype
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
delete_old_dirsbool(
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
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
¥
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	
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
¾
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
executor_typestring 
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
 "serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718¦è

MoviesEmbedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	óK*+
shared_nameMoviesEmbedding/embeddings

.MoviesEmbedding/embeddings/Read/ReadVariableOpReadVariableOpMoviesEmbedding/embeddings*
_output_shapes
:	óK*
dtype0

UsersEmbedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape:	â**
shared_nameUsersEmbedding/embeddings

-UsersEmbedding/embeddings/Read/ReadVariableOpReadVariableOpUsersEmbedding/embeddings*
_output_shapes
:	â*
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
È
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueùBö Bï

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
	loss

	variables
regularization_losses
trainable_variables
	keras_api

signatures

_init_input_shape

_init_input_shape
b

embeddings
	variables
regularization_losses
trainable_variables
	keras_api
b

embeddings
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
regularization_losses
trainable_variables
	keras_api
R
	variables
 regularization_losses
!trainable_variables
"	keras_api
R
#	variables
$regularization_losses
%trainable_variables
&	keras_api
6
'iter
	(decay
)learning_rate
*momentum
 

0
1
 

0
1
­

+layers
,layer_regularization_losses
-metrics
.non_trainable_variables

	variables
/layer_metrics
regularization_losses
trainable_variables
 
 
 
jh
VARIABLE_VALUEMoviesEmbedding/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE

0
 

0
­

0layers
1layer_regularization_losses
2metrics
3non_trainable_variables
	variables
4layer_metrics
regularization_losses
trainable_variables
ig
VARIABLE_VALUEUsersEmbedding/embeddings:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUE

0
 

0
­

5layers
6layer_regularization_losses
7metrics
8non_trainable_variables
	variables
9layer_metrics
regularization_losses
trainable_variables
 
 
 
­

:layers
;layer_regularization_losses
<metrics
=non_trainable_variables
	variables
>layer_metrics
regularization_losses
trainable_variables
 
 
 
­

?layers
@layer_regularization_losses
Ametrics
Bnon_trainable_variables
	variables
Clayer_metrics
 regularization_losses
!trainable_variables
 
 
 
­

Dlayers
Elayer_regularization_losses
Fmetrics
Gnon_trainable_variables
#	variables
Hlayer_metrics
$regularization_losses
%trainable_variables
GE
VARIABLE_VALUESGD/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUE	SGD/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
YW
VARIABLE_VALUESGD/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUESGD/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
1
0
1
2
3
4
5
6
 

I0
J1
K2
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
 
 
4
	Ltotal
	Mcount
N	variables
O	keras_api
D
	Ptotal
	Qcount
R
_fn_kwargs
S	variables
T	keras_api
D
	Utotal
	Vcount
W
_fn_kwargs
X	variables
Y	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

L0
M1

N	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

P0
Q1

S	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

U0
V1

X	variables
~
serving_default_MoviesInputPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
}
serving_default_UsersInputPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ

StatefulPartitionedCallStatefulPartitionedCallserving_default_MoviesInputserving_default_UsersInputUsersEmbedding/embeddingsMoviesEmbedding/embeddings*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *.
f)R'
%__inference_signature_wrapper_1518057
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
·
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
GPU 2J 8 *)
f$R"
 __inference__traced_save_1518337
Â
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
GPU 2J 8 *,
f'R%
#__inference__traced_restore_1518383á©
â
f
J__inference_MoviesFlatten_layer_call_and_return_conditional_losses_1517842

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã
È
@__inference_MatrixFactorizationReccomender_layer_call_fn_1517977

usersinput
moviesinput
unknown:	â
	unknown_0:	óK
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCall
usersinputmoviesinputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *d
f_R]
[__inference_MatrixFactorizationReccomender_layer_call_and_return_conditional_losses_15179602
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
UsersInput:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameMoviesInput
/
¯
[__inference_MatrixFactorizationReccomender_layer_call_and_return_conditional_losses_1517879

inputs
inputs_1)
usersembedding_1517813:	â*
moviesembedding_1517833:	óK
identity¢'MoviesEmbedding/StatefulPartitionedCall¢<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp¢&UsersEmbedding/StatefulPartitionedCall¢;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp
&UsersEmbedding/StatefulPartitionedCallStatefulPartitionedCallinputsusersembedding_1517813*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_UsersEmbedding_layer_call_and_return_conditional_losses_15178122(
&UsersEmbedding/StatefulPartitionedCall¥
'MoviesEmbedding/StatefulPartitionedCallStatefulPartitionedCallinputs_1moviesembedding_1517833*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_MoviesEmbedding_layer_call_and_return_conditional_losses_15178322)
'MoviesEmbedding/StatefulPartitionedCall
MoviesFlatten/PartitionedCallPartitionedCall0MoviesEmbedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_MoviesFlatten_layer_call_and_return_conditional_losses_15178422
MoviesFlatten/PartitionedCall
UsersFlatten/PartitionedCallPartitionedCall/UsersEmbedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_UsersFlatten_layer_call_and_return_conditional_losses_15178502
UsersFlatten/PartitionedCall£
DotProduct/PartitionedCallPartitionedCall&MoviesFlatten/PartitionedCall:output:0%UsersFlatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_DotProduct_layer_call_and_return_conditional_losses_15178642
DotProduct/PartitionedCallÕ
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpmoviesembedding_1517833*
_output_shapes
:	óK*
dtype02>
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOpØ
-MoviesEmbedding/embeddings/Regularizer/SquareSquareDMoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	óK2/
-MoviesEmbedding/embeddings/Regularizer/Square­
,MoviesEmbedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,MoviesEmbedding/embeddings/Regularizer/Constê
*MoviesEmbedding/embeddings/Regularizer/SumSum1MoviesEmbedding/embeddings/Regularizer/Square:y:05MoviesEmbedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*MoviesEmbedding/embeddings/Regularizer/Sum¡
,MoviesEmbedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752.
,MoviesEmbedding/embeddings/Regularizer/mul/xì
*MoviesEmbedding/embeddings/Regularizer/mulMul5MoviesEmbedding/embeddings/Regularizer/mul/x:output:03MoviesEmbedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*MoviesEmbedding/embeddings/Regularizer/mulÒ
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpusersembedding_1517813*
_output_shapes
:	â*
dtype02=
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOpÕ
,UsersEmbedding/embeddings/Regularizer/SquareSquareCUsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	â2.
,UsersEmbedding/embeddings/Regularizer/Square«
+UsersEmbedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+UsersEmbedding/embeddings/Regularizer/Constæ
)UsersEmbedding/embeddings/Regularizer/SumSum0UsersEmbedding/embeddings/Regularizer/Square:y:04UsersEmbedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)UsersEmbedding/embeddings/Regularizer/Sum
+UsersEmbedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752-
+UsersEmbedding/embeddings/Regularizer/mul/xè
)UsersEmbedding/embeddings/Regularizer/mulMul4UsersEmbedding/embeddings/Regularizer/mul/x:output:02UsersEmbedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)UsersEmbedding/embeddings/Regularizer/mulÇ
IdentityIdentity#DotProduct/PartitionedCall:output:0(^MoviesEmbedding/StatefulPartitionedCall=^MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp'^UsersEmbedding/StatefulPartitionedCall<^UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : 2R
'MoviesEmbedding/StatefulPartitionedCall'MoviesEmbedding/StatefulPartitionedCall2|
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp2P
&UsersEmbedding/StatefulPartitionedCall&UsersEmbedding/StatefulPartitionedCall2z
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
 

q
G__inference_DotProduct_layer_call_and_return_conditional_losses_1517864

inputs
inputs_1
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim}

ExpandDims
ExpandDimsinputsExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims_1/dim
ExpandDims_1
ExpandDimsinputs_1ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ExpandDims_1
MatMulBatchMatMulV2ExpandDims:output:0ExpandDims_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMulM
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:2
Shapew
SqueezeSqueezeMatMul:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims
2	
Squeezed
IdentityIdentitySqueeze:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
/
¶
[__inference_MatrixFactorizationReccomender_layer_call_and_return_conditional_losses_1518029

usersinput
moviesinput)
usersembedding_1518007:	â*
moviesembedding_1518010:	óK
identity¢'MoviesEmbedding/StatefulPartitionedCall¢<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp¢&UsersEmbedding/StatefulPartitionedCall¢;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp£
&UsersEmbedding/StatefulPartitionedCallStatefulPartitionedCall
usersinputusersembedding_1518007*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_UsersEmbedding_layer_call_and_return_conditional_losses_15178122(
&UsersEmbedding/StatefulPartitionedCall¨
'MoviesEmbedding/StatefulPartitionedCallStatefulPartitionedCallmoviesinputmoviesembedding_1518010*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_MoviesEmbedding_layer_call_and_return_conditional_losses_15178322)
'MoviesEmbedding/StatefulPartitionedCall
MoviesFlatten/PartitionedCallPartitionedCall0MoviesEmbedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_MoviesFlatten_layer_call_and_return_conditional_losses_15178422
MoviesFlatten/PartitionedCall
UsersFlatten/PartitionedCallPartitionedCall/UsersEmbedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_UsersFlatten_layer_call_and_return_conditional_losses_15178502
UsersFlatten/PartitionedCall£
DotProduct/PartitionedCallPartitionedCall&MoviesFlatten/PartitionedCall:output:0%UsersFlatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_DotProduct_layer_call_and_return_conditional_losses_15178642
DotProduct/PartitionedCallÕ
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpmoviesembedding_1518010*
_output_shapes
:	óK*
dtype02>
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOpØ
-MoviesEmbedding/embeddings/Regularizer/SquareSquareDMoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	óK2/
-MoviesEmbedding/embeddings/Regularizer/Square­
,MoviesEmbedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,MoviesEmbedding/embeddings/Regularizer/Constê
*MoviesEmbedding/embeddings/Regularizer/SumSum1MoviesEmbedding/embeddings/Regularizer/Square:y:05MoviesEmbedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*MoviesEmbedding/embeddings/Regularizer/Sum¡
,MoviesEmbedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752.
,MoviesEmbedding/embeddings/Regularizer/mul/xì
*MoviesEmbedding/embeddings/Regularizer/mulMul5MoviesEmbedding/embeddings/Regularizer/mul/x:output:03MoviesEmbedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*MoviesEmbedding/embeddings/Regularizer/mulÒ
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpusersembedding_1518007*
_output_shapes
:	â*
dtype02=
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOpÕ
,UsersEmbedding/embeddings/Regularizer/SquareSquareCUsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	â2.
,UsersEmbedding/embeddings/Regularizer/Square«
+UsersEmbedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+UsersEmbedding/embeddings/Regularizer/Constæ
)UsersEmbedding/embeddings/Regularizer/SumSum0UsersEmbedding/embeddings/Regularizer/Square:y:04UsersEmbedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)UsersEmbedding/embeddings/Regularizer/Sum
+UsersEmbedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752-
+UsersEmbedding/embeddings/Regularizer/mul/xè
)UsersEmbedding/embeddings/Regularizer/mulMul4UsersEmbedding/embeddings/Regularizer/mul/x:output:02UsersEmbedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)UsersEmbedding/embeddings/Regularizer/mulÇ
IdentityIdentity#DotProduct/PartitionedCall:output:0(^MoviesEmbedding/StatefulPartitionedCall=^MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp'^UsersEmbedding/StatefulPartitionedCall<^UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : 2R
'MoviesEmbedding/StatefulPartitionedCall'MoviesEmbedding/StatefulPartitionedCall2|
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp2P
&UsersEmbedding/StatefulPartitionedCall&UsersEmbedding/StatefulPartitionedCall2z
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp:S O
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
UsersInput:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameMoviesInput
/
¶
[__inference_MatrixFactorizationReccomender_layer_call_and_return_conditional_losses_1518003

usersinput
moviesinput)
usersembedding_1517981:	â*
moviesembedding_1517984:	óK
identity¢'MoviesEmbedding/StatefulPartitionedCall¢<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp¢&UsersEmbedding/StatefulPartitionedCall¢;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp£
&UsersEmbedding/StatefulPartitionedCallStatefulPartitionedCall
usersinputusersembedding_1517981*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_UsersEmbedding_layer_call_and_return_conditional_losses_15178122(
&UsersEmbedding/StatefulPartitionedCall¨
'MoviesEmbedding/StatefulPartitionedCallStatefulPartitionedCallmoviesinputmoviesembedding_1517984*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_MoviesEmbedding_layer_call_and_return_conditional_losses_15178322)
'MoviesEmbedding/StatefulPartitionedCall
MoviesFlatten/PartitionedCallPartitionedCall0MoviesEmbedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_MoviesFlatten_layer_call_and_return_conditional_losses_15178422
MoviesFlatten/PartitionedCall
UsersFlatten/PartitionedCallPartitionedCall/UsersEmbedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_UsersFlatten_layer_call_and_return_conditional_losses_15178502
UsersFlatten/PartitionedCall£
DotProduct/PartitionedCallPartitionedCall&MoviesFlatten/PartitionedCall:output:0%UsersFlatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_DotProduct_layer_call_and_return_conditional_losses_15178642
DotProduct/PartitionedCallÕ
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpmoviesembedding_1517984*
_output_shapes
:	óK*
dtype02>
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOpØ
-MoviesEmbedding/embeddings/Regularizer/SquareSquareDMoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	óK2/
-MoviesEmbedding/embeddings/Regularizer/Square­
,MoviesEmbedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,MoviesEmbedding/embeddings/Regularizer/Constê
*MoviesEmbedding/embeddings/Regularizer/SumSum1MoviesEmbedding/embeddings/Regularizer/Square:y:05MoviesEmbedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*MoviesEmbedding/embeddings/Regularizer/Sum¡
,MoviesEmbedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752.
,MoviesEmbedding/embeddings/Regularizer/mul/xì
*MoviesEmbedding/embeddings/Regularizer/mulMul5MoviesEmbedding/embeddings/Regularizer/mul/x:output:03MoviesEmbedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*MoviesEmbedding/embeddings/Regularizer/mulÒ
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpusersembedding_1517981*
_output_shapes
:	â*
dtype02=
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOpÕ
,UsersEmbedding/embeddings/Regularizer/SquareSquareCUsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	â2.
,UsersEmbedding/embeddings/Regularizer/Square«
+UsersEmbedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+UsersEmbedding/embeddings/Regularizer/Constæ
)UsersEmbedding/embeddings/Regularizer/SumSum0UsersEmbedding/embeddings/Regularizer/Square:y:04UsersEmbedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)UsersEmbedding/embeddings/Regularizer/Sum
+UsersEmbedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752-
+UsersEmbedding/embeddings/Regularizer/mul/xè
)UsersEmbedding/embeddings/Regularizer/mulMul4UsersEmbedding/embeddings/Regularizer/mul/x:output:02UsersEmbedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)UsersEmbedding/embeddings/Regularizer/mulÇ
IdentityIdentity#DotProduct/PartitionedCall:output:0(^MoviesEmbedding/StatefulPartitionedCall=^MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp'^UsersEmbedding/StatefulPartitionedCall<^UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : 2R
'MoviesEmbedding/StatefulPartitionedCall'MoviesEmbedding/StatefulPartitionedCall2|
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp2P
&UsersEmbedding/StatefulPartitionedCall&UsersEmbedding/StatefulPartitionedCall2z
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp:S O
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
UsersInput:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameMoviesInput
î
É
__inference_loss_fn_0_1518266X
Emoviesembedding_embeddings_regularizer_square_readvariableop_resource:	óK
identity¢<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpEmoviesembedding_embeddings_regularizer_square_readvariableop_resource*
_output_shapes
:	óK*
dtype02>
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOpØ
-MoviesEmbedding/embeddings/Regularizer/SquareSquareDMoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	óK2/
-MoviesEmbedding/embeddings/Regularizer/Square­
,MoviesEmbedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,MoviesEmbedding/embeddings/Regularizer/Constê
*MoviesEmbedding/embeddings/Regularizer/SumSum1MoviesEmbedding/embeddings/Regularizer/Square:y:05MoviesEmbedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*MoviesEmbedding/embeddings/Regularizer/Sum¡
,MoviesEmbedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752.
,MoviesEmbedding/embeddings/Regularizer/mul/xì
*MoviesEmbedding/embeddings/Regularizer/mulMul5MoviesEmbedding/embeddings/Regularizer/mul/x:output:03MoviesEmbedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*MoviesEmbedding/embeddings/Regularizer/mul°
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
#

 __inference__traced_save_1518337
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

identity_1¢MergeV2Checkpoints
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
Const_1
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
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*©
valueBB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names¢
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
SaveV2/shape_and_slices©
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:05savev2_moviesembedding_embeddings_read_readvariableop4savev2_usersembedding_embeddings_read_readvariableop#savev2_sgd_iter_read_readvariableop$savev2_sgd_decay_read_readvariableop,savev2_sgd_learning_rate_read_readvariableop'savev2_sgd_momentum_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
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

identity_1Identity_1:output:0*A
_input_shapes0
.: :	óK:	â: : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	óK:%!

_output_shapes
:	â:
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
Ö
Ç
__inference_loss_fn_1_1518277W
Dusersembedding_embeddings_regularizer_square_readvariableop_resource:	â
identity¢;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpDusersembedding_embeddings_regularizer_square_readvariableop_resource*
_output_shapes
:	â*
dtype02=
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOpÕ
,UsersEmbedding/embeddings/Regularizer/SquareSquareCUsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	â2.
,UsersEmbedding/embeddings/Regularizer/Square«
+UsersEmbedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+UsersEmbedding/embeddings/Regularizer/Constæ
)UsersEmbedding/embeddings/Regularizer/SumSum0UsersEmbedding/embeddings/Regularizer/Square:y:04UsersEmbedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)UsersEmbedding/embeddings/Regularizer/Sum
+UsersEmbedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752-
+UsersEmbedding/embeddings/Regularizer/mul/xè
)UsersEmbedding/embeddings/Regularizer/mulMul4UsersEmbedding/embeddings/Regularizer/mul/x:output:02UsersEmbedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)UsersEmbedding/embeddings/Regularizer/mul®
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
/
¯
[__inference_MatrixFactorizationReccomender_layer_call_and_return_conditional_losses_1517960

inputs
inputs_1)
usersembedding_1517938:	â*
moviesembedding_1517941:	óK
identity¢'MoviesEmbedding/StatefulPartitionedCall¢<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp¢&UsersEmbedding/StatefulPartitionedCall¢;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp
&UsersEmbedding/StatefulPartitionedCallStatefulPartitionedCallinputsusersembedding_1517938*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_UsersEmbedding_layer_call_and_return_conditional_losses_15178122(
&UsersEmbedding/StatefulPartitionedCall¥
'MoviesEmbedding/StatefulPartitionedCallStatefulPartitionedCallinputs_1moviesembedding_1517941*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_MoviesEmbedding_layer_call_and_return_conditional_losses_15178322)
'MoviesEmbedding/StatefulPartitionedCall
MoviesFlatten/PartitionedCallPartitionedCall0MoviesEmbedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_MoviesFlatten_layer_call_and_return_conditional_losses_15178422
MoviesFlatten/PartitionedCall
UsersFlatten/PartitionedCallPartitionedCall/UsersEmbedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_UsersFlatten_layer_call_and_return_conditional_losses_15178502
UsersFlatten/PartitionedCall£
DotProduct/PartitionedCallPartitionedCall&MoviesFlatten/PartitionedCall:output:0%UsersFlatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_DotProduct_layer_call_and_return_conditional_losses_15178642
DotProduct/PartitionedCallÕ
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpmoviesembedding_1517941*
_output_shapes
:	óK*
dtype02>
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOpØ
-MoviesEmbedding/embeddings/Regularizer/SquareSquareDMoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	óK2/
-MoviesEmbedding/embeddings/Regularizer/Square­
,MoviesEmbedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,MoviesEmbedding/embeddings/Regularizer/Constê
*MoviesEmbedding/embeddings/Regularizer/SumSum1MoviesEmbedding/embeddings/Regularizer/Square:y:05MoviesEmbedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*MoviesEmbedding/embeddings/Regularizer/Sum¡
,MoviesEmbedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752.
,MoviesEmbedding/embeddings/Regularizer/mul/xì
*MoviesEmbedding/embeddings/Regularizer/mulMul5MoviesEmbedding/embeddings/Regularizer/mul/x:output:03MoviesEmbedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*MoviesEmbedding/embeddings/Regularizer/mulÒ
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpusersembedding_1517938*
_output_shapes
:	â*
dtype02=
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOpÕ
,UsersEmbedding/embeddings/Regularizer/SquareSquareCUsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	â2.
,UsersEmbedding/embeddings/Regularizer/Square«
+UsersEmbedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+UsersEmbedding/embeddings/Regularizer/Constæ
)UsersEmbedding/embeddings/Regularizer/SumSum0UsersEmbedding/embeddings/Regularizer/Square:y:04UsersEmbedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)UsersEmbedding/embeddings/Regularizer/Sum
+UsersEmbedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752-
+UsersEmbedding/embeddings/Regularizer/mul/xè
)UsersEmbedding/embeddings/Regularizer/mulMul4UsersEmbedding/embeddings/Regularizer/mul/x:output:02UsersEmbedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)UsersEmbedding/embeddings/Regularizer/mulÇ
IdentityIdentity#DotProduct/PartitionedCall:output:0(^MoviesEmbedding/StatefulPartitionedCall=^MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp'^UsersEmbedding/StatefulPartitionedCall<^UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : 2R
'MoviesEmbedding/StatefulPartitionedCall'MoviesEmbedding/StatefulPartitionedCall2|
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp2P
&UsersEmbedding/StatefulPartitionedCall&UsersEmbedding/StatefulPartitionedCall2z
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ô
Ã
@__inference_MatrixFactorizationReccomender_layer_call_fn_1518077
inputs_0
inputs_1
unknown:	â
	unknown_0:	óK
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *d
f_R]
[__inference_MatrixFactorizationReccomender_layer_call_and_return_conditional_losses_15179602
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
Ô
Ã
@__inference_MatrixFactorizationReccomender_layer_call_fn_1518067
inputs_0
inputs_1
unknown:	â
	unknown_0:	óK
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *d
f_R]
[__inference_MatrixFactorizationReccomender_layer_call_and_return_conditional_losses_15178792
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
¨

s
G__inference_DotProduct_layer_call_and_return_conditional_losses_1518255
inputs_0
inputs_1
identityb
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims/dim

ExpandDims
ExpandDimsinputs_0ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

ExpandDimsf
ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2
ExpandDims_1/dim
ExpandDims_1
ExpandDimsinputs_1ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
ExpandDims_1
MatMulBatchMatMulV2ExpandDims:output:0ExpandDims_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMulM
ShapeShapeMatMul:output:0*
T0*
_output_shapes
:2
Shapew
SqueezeSqueezeMatMul:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims
2	
Squeezed
IdentityIdentitySqueeze:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
é
è
K__inference_UsersEmbedding_layer_call_and_return_conditional_losses_1517812

inputs+
embedding_lookup_1517800:	â
identity¢;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp¢embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Castÿ
embedding_lookupResourceGatherembedding_lookup_1517800Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*+
_class!
loc:@embedding_lookup/1517800*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_lookupî
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@embedding_lookup/1517800*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity 
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity_1Ô
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_lookup_1517800*
_output_shapes
:	â*
dtype02=
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOpÕ
,UsersEmbedding/embeddings/Regularizer/SquareSquareCUsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	â2.
,UsersEmbedding/embeddings/Regularizer/Square«
+UsersEmbedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+UsersEmbedding/embeddings/Regularizer/Constæ
)UsersEmbedding/embeddings/Regularizer/SumSum0UsersEmbedding/embeddings/Regularizer/Square:y:04UsersEmbedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)UsersEmbedding/embeddings/Regularizer/Sum
+UsersEmbedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752-
+UsersEmbedding/embeddings/Regularizer/mul/xè
)UsersEmbedding/embeddings/Regularizer/mulMul4UsersEmbedding/embeddings/Regularizer/mul/x:output:02UsersEmbedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)UsersEmbedding/embeddings/Regularizer/mulÍ
IdentityIdentity$embedding_lookup/Identity_1:output:0<^UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp^embedding_lookup*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2z
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


0__inference_UsersEmbedding_layer_call_fn_1518199

inputs
unknown:	â
identity¢StatefulPartitionedCallò
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *T
fORM
K__inference_UsersEmbedding_layer_call_and_return_conditional_losses_15178122
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÿ
ê
L__inference_MoviesEmbedding_layer_call_and_return_conditional_losses_1517832

inputs+
embedding_lookup_1517820:	óK
identity¢<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp¢embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Castÿ
embedding_lookupResourceGatherembedding_lookup_1517820Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*+
_class!
loc:@embedding_lookup/1517820*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_lookupî
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@embedding_lookup/1517820*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity 
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity_1Ö
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_lookup_1517820*
_output_shapes
:	óK*
dtype02>
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOpØ
-MoviesEmbedding/embeddings/Regularizer/SquareSquareDMoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	óK2/
-MoviesEmbedding/embeddings/Regularizer/Square­
,MoviesEmbedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,MoviesEmbedding/embeddings/Regularizer/Constê
*MoviesEmbedding/embeddings/Regularizer/SumSum1MoviesEmbedding/embeddings/Regularizer/Square:y:05MoviesEmbedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*MoviesEmbedding/embeddings/Regularizer/Sum¡
,MoviesEmbedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752.
,MoviesEmbedding/embeddings/Regularizer/mul/xì
*MoviesEmbedding/embeddings/Regularizer/mulMul5MoviesEmbedding/embeddings/Regularizer/mul/x:output:03MoviesEmbedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*MoviesEmbedding/embeddings/Regularizer/mulÎ
IdentityIdentity$embedding_lookup/Identity_1:output:0=^MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp^embedding_lookup*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2|
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÿ
ê
L__inference_MoviesEmbedding_layer_call_and_return_conditional_losses_1518186

inputs+
embedding_lookup_1518174:	óK
identity¢<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp¢embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Castÿ
embedding_lookupResourceGatherembedding_lookup_1518174Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*+
_class!
loc:@embedding_lookup/1518174*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_lookupî
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@embedding_lookup/1518174*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity 
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity_1Ö
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_lookup_1518174*
_output_shapes
:	óK*
dtype02>
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOpØ
-MoviesEmbedding/embeddings/Regularizer/SquareSquareDMoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	óK2/
-MoviesEmbedding/embeddings/Regularizer/Square­
,MoviesEmbedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,MoviesEmbedding/embeddings/Regularizer/Constê
*MoviesEmbedding/embeddings/Regularizer/SumSum1MoviesEmbedding/embeddings/Regularizer/Square:y:05MoviesEmbedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*MoviesEmbedding/embeddings/Regularizer/Sum¡
,MoviesEmbedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752.
,MoviesEmbedding/embeddings/Regularizer/mul/xì
*MoviesEmbedding/embeddings/Regularizer/mulMul5MoviesEmbedding/embeddings/Regularizer/mul/x:output:03MoviesEmbedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*MoviesEmbedding/embeddings/Regularizer/mulÎ
IdentityIdentity$embedding_lookup/Identity_1:output:0=^MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp^embedding_lookup*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2|
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Î
J
.__inference_UsersFlatten_layer_call_fn_1518231

inputs
identityÇ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_UsersFlatten_layer_call_and_return_conditional_losses_15178502
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ð
K
/__inference_MoviesFlatten_layer_call_fn_1518220

inputs
identityÈ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *S
fNRL
J__inference_MoviesFlatten_layer_call_and_return_conditional_losses_15178422
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
á
e
I__inference_UsersFlatten_layer_call_and_return_conditional_losses_1517850

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
á
e
I__inference_UsersFlatten_layer_call_and_return_conditional_losses_1518237

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ã
È
@__inference_MatrixFactorizationReccomender_layer_call_fn_1517886

usersinput
moviesinput
unknown:	â
	unknown_0:	óK
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCall
usersinputmoviesinputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *d
f_R]
[__inference_MatrixFactorizationReccomender_layer_call_and_return_conditional_losses_15178792
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
UsersInput:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameMoviesInput

­
%__inference_signature_wrapper_1518057
moviesinput

usersinput
unknown:	â
	unknown_0:	óK
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCall
usersinputmoviesinputunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__wrapped_model_15177872
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameMoviesInput:SO
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
UsersInput
È;
Å
[__inference_MatrixFactorizationReccomender_layer_call_and_return_conditional_losses_1518117
inputs_0
inputs_1:
'usersembedding_embedding_lookup_1518082:	â;
(moviesembedding_embedding_lookup_1518088:	óK
identity¢ MoviesEmbedding/embedding_lookup¢<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp¢UsersEmbedding/embedding_lookup¢;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp}
UsersEmbedding/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
UsersEmbedding/CastÊ
UsersEmbedding/embedding_lookupResourceGather'usersembedding_embedding_lookup_1518082UsersEmbedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*:
_class0
.,loc:@UsersEmbedding/embedding_lookup/1518082*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02!
UsersEmbedding/embedding_lookupª
(UsersEmbedding/embedding_lookup/IdentityIdentity(UsersEmbedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*:
_class0
.,loc:@UsersEmbedding/embedding_lookup/1518082*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(UsersEmbedding/embedding_lookup/IdentityÍ
*UsersEmbedding/embedding_lookup/Identity_1Identity1UsersEmbedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*UsersEmbedding/embedding_lookup/Identity_1
MoviesEmbedding/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MoviesEmbedding/CastÏ
 MoviesEmbedding/embedding_lookupResourceGather(moviesembedding_embedding_lookup_1518088MoviesEmbedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*;
_class1
/-loc:@MoviesEmbedding/embedding_lookup/1518088*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02"
 MoviesEmbedding/embedding_lookup®
)MoviesEmbedding/embedding_lookup/IdentityIdentity)MoviesEmbedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*;
_class1
/-loc:@MoviesEmbedding/embedding_lookup/1518088*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)MoviesEmbedding/embedding_lookup/IdentityÐ
+MoviesEmbedding/embedding_lookup/Identity_1Identity2MoviesEmbedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+MoviesEmbedding/embedding_lookup/Identity_1{
MoviesFlatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
MoviesFlatten/Const¿
MoviesFlatten/ReshapeReshape4MoviesEmbedding/embedding_lookup/Identity_1:output:0MoviesFlatten/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MoviesFlatten/Reshapey
UsersFlatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
UsersFlatten/Const»
UsersFlatten/ReshapeReshape3UsersEmbedding/embedding_lookup/Identity_1:output:0UsersFlatten/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
UsersFlatten/Reshapex
DotProduct/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
DotProduct/ExpandDims/dim¶
DotProduct/ExpandDims
ExpandDimsMoviesFlatten/Reshape:output:0"DotProduct/ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
DotProduct/ExpandDims|
DotProduct/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2
DotProduct/ExpandDims_1/dim»
DotProduct/ExpandDims_1
ExpandDimsUsersFlatten/Reshape:output:0$DotProduct/ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
DotProduct/ExpandDims_1¯
DotProduct/MatMulBatchMatMulV2DotProduct/ExpandDims:output:0 DotProduct/ExpandDims_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
DotProduct/MatMuln
DotProduct/ShapeShapeDotProduct/MatMul:output:0*
T0*
_output_shapes
:2
DotProduct/Shape
DotProduct/SqueezeSqueezeDotProduct/MatMul:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims
2
DotProduct/Squeezeæ
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp(moviesembedding_embedding_lookup_1518088*
_output_shapes
:	óK*
dtype02>
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOpØ
-MoviesEmbedding/embeddings/Regularizer/SquareSquareDMoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	óK2/
-MoviesEmbedding/embeddings/Regularizer/Square­
,MoviesEmbedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,MoviesEmbedding/embeddings/Regularizer/Constê
*MoviesEmbedding/embeddings/Regularizer/SumSum1MoviesEmbedding/embeddings/Regularizer/Square:y:05MoviesEmbedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*MoviesEmbedding/embeddings/Regularizer/Sum¡
,MoviesEmbedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752.
,MoviesEmbedding/embeddings/Regularizer/mul/xì
*MoviesEmbedding/embeddings/Regularizer/mulMul5MoviesEmbedding/embeddings/Regularizer/mul/x:output:03MoviesEmbedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*MoviesEmbedding/embeddings/Regularizer/mulã
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp'usersembedding_embedding_lookup_1518082*
_output_shapes
:	â*
dtype02=
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOpÕ
,UsersEmbedding/embeddings/Regularizer/SquareSquareCUsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	â2.
,UsersEmbedding/embeddings/Regularizer/Square«
+UsersEmbedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+UsersEmbedding/embeddings/Regularizer/Constæ
)UsersEmbedding/embeddings/Regularizer/SumSum0UsersEmbedding/embeddings/Regularizer/Square:y:04UsersEmbedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)UsersEmbedding/embeddings/Regularizer/Sum
+UsersEmbedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752-
+UsersEmbedding/embeddings/Regularizer/mul/xè
)UsersEmbedding/embeddings/Regularizer/mulMul4UsersEmbedding/embeddings/Regularizer/mul/x:output:02UsersEmbedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)UsersEmbedding/embeddings/Regularizer/mul±
IdentityIdentityDotProduct/Squeeze:output:0!^MoviesEmbedding/embedding_lookup=^MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp ^UsersEmbedding/embedding_lookup<^UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : 2D
 MoviesEmbedding/embedding_lookup MoviesEmbedding/embedding_lookup2|
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp2B
UsersEmbedding/embedding_lookupUsersEmbedding/embedding_lookup2z
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
Ç
X
,__inference_DotProduct_layer_call_fn_1518243
inputs_0
inputs_1
identityÒ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_DotProduct_layer_call_and_return_conditional_losses_15178642
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
5
Ó
#__inference__traced_restore_1518383
file_prefix>
+assignvariableop_moviesembedding_embeddings:	óK?
,assignvariableop_1_usersembedding_embeddings:	â%
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
identity_13¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*©
valueBB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names¨
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*-
value$B"B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesì
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

Identityª
AssignVariableOpAssignVariableOp+assignvariableop_moviesembedding_embeddingsIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1±
AssignVariableOp_1AssignVariableOp,assignvariableop_1_usersembedding_embeddingsIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2 
AssignVariableOp_2AssignVariableOpassignvariableop_2_sgd_iterIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¡
AssignVariableOp_3AssignVariableOpassignvariableop_3_sgd_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4©
AssignVariableOp_4AssignVariableOp$assignvariableop_4_sgd_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¤
AssignVariableOp_5AssignVariableOpassignvariableop_5_sgd_momentumIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6
AssignVariableOp_6AssignVariableOpassignvariableop_6_totalIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7
AssignVariableOp_7AssignVariableOpassignvariableop_7_countIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8
AssignVariableOp_8AssignVariableOpassignvariableop_8_total_1Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9
AssignVariableOp_9AssignVariableOpassignvariableop_9_count_1Identity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10£
AssignVariableOp_10AssignVariableOpassignvariableop_10_total_2Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11£
AssignVariableOp_11AssignVariableOpassignvariableop_11_count_2Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_119
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpæ
Identity_12Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_12Ù
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
´5

"__inference__wrapped_model_1517787

usersinput
moviesinputY
Fmatrixfactorizationreccomender_usersembedding_embedding_lookup_1517764:	âZ
Gmatrixfactorizationreccomender_moviesembedding_embedding_lookup_1517770:	óK
identity¢?MatrixFactorizationReccomender/MoviesEmbedding/embedding_lookup¢>MatrixFactorizationReccomender/UsersEmbedding/embedding_lookup½
2MatrixFactorizationReccomender/UsersEmbedding/CastCast
usersinput*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2MatrixFactorizationReccomender/UsersEmbedding/Castå
>MatrixFactorizationReccomender/UsersEmbedding/embedding_lookupResourceGatherFmatrixfactorizationreccomender_usersembedding_embedding_lookup_15177646MatrixFactorizationReccomender/UsersEmbedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*Y
_classO
MKloc:@MatrixFactorizationReccomender/UsersEmbedding/embedding_lookup/1517764*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02@
>MatrixFactorizationReccomender/UsersEmbedding/embedding_lookup¦
GMatrixFactorizationReccomender/UsersEmbedding/embedding_lookup/IdentityIdentityGMatrixFactorizationReccomender/UsersEmbedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*Y
_classO
MKloc:@MatrixFactorizationReccomender/UsersEmbedding/embedding_lookup/1517764*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2I
GMatrixFactorizationReccomender/UsersEmbedding/embedding_lookup/Identityª
IMatrixFactorizationReccomender/UsersEmbedding/embedding_lookup/Identity_1IdentityPMatrixFactorizationReccomender/UsersEmbedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2K
IMatrixFactorizationReccomender/UsersEmbedding/embedding_lookup/Identity_1À
3MatrixFactorizationReccomender/MoviesEmbedding/CastCastmoviesinput*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ25
3MatrixFactorizationReccomender/MoviesEmbedding/Castê
?MatrixFactorizationReccomender/MoviesEmbedding/embedding_lookupResourceGatherGmatrixfactorizationreccomender_moviesembedding_embedding_lookup_15177707MatrixFactorizationReccomender/MoviesEmbedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*Z
_classP
NLloc:@MatrixFactorizationReccomender/MoviesEmbedding/embedding_lookup/1517770*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02A
?MatrixFactorizationReccomender/MoviesEmbedding/embedding_lookupª
HMatrixFactorizationReccomender/MoviesEmbedding/embedding_lookup/IdentityIdentityHMatrixFactorizationReccomender/MoviesEmbedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*Z
_classP
NLloc:@MatrixFactorizationReccomender/MoviesEmbedding/embedding_lookup/1517770*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2J
HMatrixFactorizationReccomender/MoviesEmbedding/embedding_lookup/Identity­
JMatrixFactorizationReccomender/MoviesEmbedding/embedding_lookup/Identity_1IdentityQMatrixFactorizationReccomender/MoviesEmbedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2L
JMatrixFactorizationReccomender/MoviesEmbedding/embedding_lookup/Identity_1¹
2MatrixFactorizationReccomender/MoviesFlatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   24
2MatrixFactorizationReccomender/MoviesFlatten/Const»
4MatrixFactorizationReccomender/MoviesFlatten/ReshapeReshapeSMatrixFactorizationReccomender/MoviesEmbedding/embedding_lookup/Identity_1:output:0;MatrixFactorizationReccomender/MoviesFlatten/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4MatrixFactorizationReccomender/MoviesFlatten/Reshape·
1MatrixFactorizationReccomender/UsersFlatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   23
1MatrixFactorizationReccomender/UsersFlatten/Const·
3MatrixFactorizationReccomender/UsersFlatten/ReshapeReshapeRMatrixFactorizationReccomender/UsersEmbedding/embedding_lookup/Identity_1:output:0:MatrixFactorizationReccomender/UsersFlatten/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ25
3MatrixFactorizationReccomender/UsersFlatten/Reshape¶
8MatrixFactorizationReccomender/DotProduct/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2:
8MatrixFactorizationReccomender/DotProduct/ExpandDims/dim²
4MatrixFactorizationReccomender/DotProduct/ExpandDims
ExpandDims=MatrixFactorizationReccomender/MoviesFlatten/Reshape:output:0AMatrixFactorizationReccomender/DotProduct/ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ26
4MatrixFactorizationReccomender/DotProduct/ExpandDimsº
:MatrixFactorizationReccomender/DotProduct/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2<
:MatrixFactorizationReccomender/DotProduct/ExpandDims_1/dim·
6MatrixFactorizationReccomender/DotProduct/ExpandDims_1
ExpandDims<MatrixFactorizationReccomender/UsersFlatten/Reshape:output:0CMatrixFactorizationReccomender/DotProduct/ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ28
6MatrixFactorizationReccomender/DotProduct/ExpandDims_1«
0MatrixFactorizationReccomender/DotProduct/MatMulBatchMatMulV2=MatrixFactorizationReccomender/DotProduct/ExpandDims:output:0?MatrixFactorizationReccomender/DotProduct/ExpandDims_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0MatrixFactorizationReccomender/DotProduct/MatMulË
/MatrixFactorizationReccomender/DotProduct/ShapeShape9MatrixFactorizationReccomender/DotProduct/MatMul:output:0*
T0*
_output_shapes
:21
/MatrixFactorizationReccomender/DotProduct/Shapeõ
1MatrixFactorizationReccomender/DotProduct/SqueezeSqueeze9MatrixFactorizationReccomender/DotProduct/MatMul:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims
23
1MatrixFactorizationReccomender/DotProduct/Squeeze
IdentityIdentity:MatrixFactorizationReccomender/DotProduct/Squeeze:output:0@^MatrixFactorizationReccomender/MoviesEmbedding/embedding_lookup?^MatrixFactorizationReccomender/UsersEmbedding/embedding_lookup*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : 2
?MatrixFactorizationReccomender/MoviesEmbedding/embedding_lookup?MatrixFactorizationReccomender/MoviesEmbedding/embedding_lookup2
>MatrixFactorizationReccomender/UsersEmbedding/embedding_lookup>MatrixFactorizationReccomender/UsersEmbedding/embedding_lookup:S O
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$
_user_specified_name
UsersInput:TP
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%
_user_specified_nameMoviesInput
È;
Å
[__inference_MatrixFactorizationReccomender_layer_call_and_return_conditional_losses_1518157
inputs_0
inputs_1:
'usersembedding_embedding_lookup_1518122:	â;
(moviesembedding_embedding_lookup_1518128:	óK
identity¢ MoviesEmbedding/embedding_lookup¢<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp¢UsersEmbedding/embedding_lookup¢;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp}
UsersEmbedding/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
UsersEmbedding/CastÊ
UsersEmbedding/embedding_lookupResourceGather'usersembedding_embedding_lookup_1518122UsersEmbedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*:
_class0
.,loc:@UsersEmbedding/embedding_lookup/1518122*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02!
UsersEmbedding/embedding_lookupª
(UsersEmbedding/embedding_lookup/IdentityIdentity(UsersEmbedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*:
_class0
.,loc:@UsersEmbedding/embedding_lookup/1518122*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(UsersEmbedding/embedding_lookup/IdentityÍ
*UsersEmbedding/embedding_lookup/Identity_1Identity1UsersEmbedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*UsersEmbedding/embedding_lookup/Identity_1
MoviesEmbedding/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MoviesEmbedding/CastÏ
 MoviesEmbedding/embedding_lookupResourceGather(moviesembedding_embedding_lookup_1518128MoviesEmbedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*;
_class1
/-loc:@MoviesEmbedding/embedding_lookup/1518128*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02"
 MoviesEmbedding/embedding_lookup®
)MoviesEmbedding/embedding_lookup/IdentityIdentity)MoviesEmbedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*;
_class1
/-loc:@MoviesEmbedding/embedding_lookup/1518128*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2+
)MoviesEmbedding/embedding_lookup/IdentityÐ
+MoviesEmbedding/embedding_lookup/Identity_1Identity2MoviesEmbedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+MoviesEmbedding/embedding_lookup/Identity_1{
MoviesFlatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
MoviesFlatten/Const¿
MoviesFlatten/ReshapeReshape4MoviesEmbedding/embedding_lookup/Identity_1:output:0MoviesFlatten/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MoviesFlatten/Reshapey
UsersFlatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
UsersFlatten/Const»
UsersFlatten/ReshapeReshape3UsersEmbedding/embedding_lookup/Identity_1:output:0UsersFlatten/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
UsersFlatten/Reshapex
DotProduct/ExpandDims/dimConst*
_output_shapes
: *
dtype0*
value	B :2
DotProduct/ExpandDims/dim¶
DotProduct/ExpandDims
ExpandDimsMoviesFlatten/Reshape:output:0"DotProduct/ExpandDims/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
DotProduct/ExpandDims|
DotProduct/ExpandDims_1/dimConst*
_output_shapes
: *
dtype0*
value	B :2
DotProduct/ExpandDims_1/dim»
DotProduct/ExpandDims_1
ExpandDimsUsersFlatten/Reshape:output:0$DotProduct/ExpandDims_1/dim:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
DotProduct/ExpandDims_1¯
DotProduct/MatMulBatchMatMulV2DotProduct/ExpandDims:output:0 DotProduct/ExpandDims_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
DotProduct/MatMuln
DotProduct/ShapeShapeDotProduct/MatMul:output:0*
T0*
_output_shapes
:2
DotProduct/Shape
DotProduct/SqueezeSqueezeDotProduct/MatMul:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
squeeze_dims
2
DotProduct/Squeezeæ
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp(moviesembedding_embedding_lookup_1518128*
_output_shapes
:	óK*
dtype02>
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOpØ
-MoviesEmbedding/embeddings/Regularizer/SquareSquareDMoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	óK2/
-MoviesEmbedding/embeddings/Regularizer/Square­
,MoviesEmbedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2.
,MoviesEmbedding/embeddings/Regularizer/Constê
*MoviesEmbedding/embeddings/Regularizer/SumSum1MoviesEmbedding/embeddings/Regularizer/Square:y:05MoviesEmbedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2,
*MoviesEmbedding/embeddings/Regularizer/Sum¡
,MoviesEmbedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752.
,MoviesEmbedding/embeddings/Regularizer/mul/xì
*MoviesEmbedding/embeddings/Regularizer/mulMul5MoviesEmbedding/embeddings/Regularizer/mul/x:output:03MoviesEmbedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*MoviesEmbedding/embeddings/Regularizer/mulã
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOp'usersembedding_embedding_lookup_1518122*
_output_shapes
:	â*
dtype02=
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOpÕ
,UsersEmbedding/embeddings/Regularizer/SquareSquareCUsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	â2.
,UsersEmbedding/embeddings/Regularizer/Square«
+UsersEmbedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+UsersEmbedding/embeddings/Regularizer/Constæ
)UsersEmbedding/embeddings/Regularizer/SumSum0UsersEmbedding/embeddings/Regularizer/Square:y:04UsersEmbedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)UsersEmbedding/embeddings/Regularizer/Sum
+UsersEmbedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752-
+UsersEmbedding/embeddings/Regularizer/mul/xè
)UsersEmbedding/embeddings/Regularizer/mulMul4UsersEmbedding/embeddings/Regularizer/mul/x:output:02UsersEmbedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)UsersEmbedding/embeddings/Regularizer/mul±
IdentityIdentityDotProduct/Squeeze:output:0!^MoviesEmbedding/embedding_lookup=^MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp ^UsersEmbedding/embedding_lookup<^UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*=
_input_shapes,
*:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : 2D
 MoviesEmbedding/embedding_lookup MoviesEmbedding/embedding_lookup2|
<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp<MoviesEmbedding/embeddings/Regularizer/Square/ReadVariableOp2B
UsersEmbedding/embedding_lookupUsersEmbedding/embedding_lookup2z
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
é
è
K__inference_UsersEmbedding_layer_call_and_return_conditional_losses_1518215

inputs+
embedding_lookup_1518203:	â
identity¢;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp¢embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Castÿ
embedding_lookupResourceGatherembedding_lookup_1518203Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*+
_class!
loc:@embedding_lookup/1518203*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_lookupî
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@embedding_lookup/1518203*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity 
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity_1Ô
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_lookup_1518203*
_output_shapes
:	â*
dtype02=
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOpÕ
,UsersEmbedding/embeddings/Regularizer/SquareSquareCUsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:	â2.
,UsersEmbedding/embeddings/Regularizer/Square«
+UsersEmbedding/embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2-
+UsersEmbedding/embeddings/Regularizer/Constæ
)UsersEmbedding/embeddings/Regularizer/SumSum0UsersEmbedding/embeddings/Regularizer/Square:y:04UsersEmbedding/embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2+
)UsersEmbedding/embeddings/Regularizer/Sum
+UsersEmbedding/embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *½752-
+UsersEmbedding/embeddings/Regularizer/mul/xè
)UsersEmbedding/embeddings/Regularizer/mulMul4UsersEmbedding/embeddings/Regularizer/mul/x:output:02UsersEmbedding/embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2+
)UsersEmbedding/embeddings/Regularizer/mulÍ
IdentityIdentity$embedding_lookup/Identity_1:output:0<^UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp^embedding_lookup*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 2z
;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp;UsersEmbedding/embeddings/Regularizer/Square/ReadVariableOp2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


1__inference_MoviesEmbedding_layer_call_fn_1518170

inputs
unknown:	óK
identity¢StatefulPartitionedCalló
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_MoviesEmbedding_layer_call_and_return_conditional_losses_15178322
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
â
f
J__inference_MoviesFlatten_layer_call_and_return_conditional_losses_1518226

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"ÌL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ø
serving_defaultä
C
MoviesInput4
serving_default_MoviesInput:0ÿÿÿÿÿÿÿÿÿ
A

UsersInput3
serving_default_UsersInput:0ÿÿÿÿÿÿÿÿÿ>

DotProduct0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:Í
¬:
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
	loss

	variables
regularization_losses
trainable_variables
	keras_api

signatures
Z__call__
[_default_save_signature
*\&call_and_return_all_conditional_losses"È7
_tf_keras_network¬7{"name": "MatrixFactorizationReccomender", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "MatrixFactorizationReccomender", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "MoviesInput"}, "name": "MoviesInput", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "UsersInput"}, "name": "UsersInput", "inbound_nodes": []}, {"class_name": "Embedding", "config": {"name": "MoviesEmbedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 9715, "output_dim": 19, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": {"class_name": "L2", "config": {"l2": 9.999999974752427e-07}}, "activity_regularizer": null, "embeddings_constraint": {"class_name": "NonNeg", "config": {}}, "mask_zero": false, "input_length": null}, "name": "MoviesEmbedding", "inbound_nodes": [[["MoviesInput", 0, 0, {}]]]}, {"class_name": "Embedding", "config": {"name": "UsersEmbedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 610, "output_dim": 19, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}}, "embeddings_regularizer": {"class_name": "L2", "config": {"l2": 9.999999974752427e-07}}, "activity_regularizer": null, "embeddings_constraint": {"class_name": "NonNeg", "config": {}}, "mask_zero": false, "input_length": null}, "name": "UsersEmbedding", "inbound_nodes": [[["UsersInput", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "MoviesFlatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "MoviesFlatten", "inbound_nodes": [[["MoviesEmbedding", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "UsersFlatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "UsersFlatten", "inbound_nodes": [[["UsersEmbedding", 0, 0, {}]]]}, {"class_name": "Dot", "config": {"name": "DotProduct", "trainable": true, "dtype": "float32", "axes": 1, "normalize": false}, "name": "DotProduct", "inbound_nodes": [[["MoviesFlatten", 0, 0, {}], ["UsersFlatten", 0, 0, {}]]]}], "input_layers": [["UsersInput", 0, 0], ["MoviesInput", 0, 0]], "output_layers": [["DotProduct", 0, 0]]}, "shared_object_id": 13, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}], "is_graph_network": true, "save_spec": [{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "UsersInput"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "MoviesInput"]}], "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "MatrixFactorizationReccomender", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "MoviesInput"}, "name": "MoviesInput", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "UsersInput"}, "name": "UsersInput", "inbound_nodes": [], "shared_object_id": 1}, {"class_name": "Embedding", "config": {"name": "MoviesEmbedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 9715, "output_dim": 19, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 2}, "embeddings_regularizer": {"class_name": "L2", "config": {"l2": 9.999999974752427e-07}, "shared_object_id": 3}, "activity_regularizer": null, "embeddings_constraint": {"class_name": "NonNeg", "config": {}, "shared_object_id": 4}, "mask_zero": false, "input_length": null}, "name": "MoviesEmbedding", "inbound_nodes": [[["MoviesInput", 0, 0, {}]]], "shared_object_id": 5}, {"class_name": "Embedding", "config": {"name": "UsersEmbedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 610, "output_dim": 19, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 6}, "embeddings_regularizer": {"class_name": "L2", "config": {"l2": 9.999999974752427e-07}, "shared_object_id": 7}, "activity_regularizer": null, "embeddings_constraint": {"class_name": "NonNeg", "config": {}, "shared_object_id": 8}, "mask_zero": false, "input_length": null}, "name": "UsersEmbedding", "inbound_nodes": [[["UsersInput", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "Flatten", "config": {"name": "MoviesFlatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "MoviesFlatten", "inbound_nodes": [[["MoviesEmbedding", 0, 0, {}]]], "shared_object_id": 10}, {"class_name": "Flatten", "config": {"name": "UsersFlatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "UsersFlatten", "inbound_nodes": [[["UsersEmbedding", 0, 0, {}]]], "shared_object_id": 11}, {"class_name": "Dot", "config": {"name": "DotProduct", "trainable": true, "dtype": "float32", "axes": 1, "normalize": false}, "name": "DotProduct", "inbound_nodes": [[["MoviesFlatten", 0, 0, {}], ["UsersFlatten", 0, 0, {}]]], "shared_object_id": 12}], "input_layers": [["UsersInput", 0, 0], ["MoviesInput", 0, 0]], "output_layers": [["DotProduct", 0, 0]]}}, "training_config": {"loss": null, "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}, "shared_object_id": 16}, {"class_name": "MeanMetricWrapper", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}, "shared_object_id": 17}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "SGD", "config": {"name": "SGD", "learning_rate": 3.000000026176508e-09, "decay": 0.0, "momentum": 0.0, "nesterov": false}}}}

_init_input_shape"î
_tf_keras_input_layerÎ{"class_name": "InputLayer", "name": "MoviesInput", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "MoviesInput"}}

_init_input_shape"ì
_tf_keras_input_layerÌ{"class_name": "InputLayer", "name": "UsersInput", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "UsersInput"}}
	

embeddings
	variables
regularization_losses
trainable_variables
	keras_api
]__call__
*^&call_and_return_all_conditional_losses"ý
_tf_keras_layerã{"name": "MoviesEmbedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "class_name": "Embedding", "config": {"name": "MoviesEmbedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 9715, "output_dim": 19, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 2}, "embeddings_regularizer": {"class_name": "L2", "config": {"l2": 9.999999974752427e-07}, "shared_object_id": 3}, "activity_regularizer": null, "embeddings_constraint": {"class_name": "NonNeg", "config": {}, "shared_object_id": 4}, "mask_zero": false, "input_length": null}, "inbound_nodes": [[["MoviesInput", 0, 0, {}]]], "shared_object_id": 5, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
	

embeddings
	variables
regularization_losses
trainable_variables
	keras_api
___call__
*`&call_and_return_all_conditional_losses"ù
_tf_keras_layerß{"name": "UsersEmbedding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": false, "class_name": "Embedding", "config": {"name": "UsersEmbedding", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "input_dim": 610, "output_dim": 19, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.05, "maxval": 0.05, "seed": null}, "shared_object_id": 6}, "embeddings_regularizer": {"class_name": "L2", "config": {"l2": 9.999999974752427e-07}, "shared_object_id": 7}, "activity_regularizer": null, "embeddings_constraint": {"class_name": "NonNeg", "config": {}, "shared_object_id": 8}, "mask_zero": false, "input_length": null}, "inbound_nodes": [[["UsersInput", 0, 0, {}]]], "shared_object_id": 9, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
Ò
	variables
regularization_losses
trainable_variables
	keras_api
a__call__
*b&call_and_return_all_conditional_losses"Ã
_tf_keras_layer©{"name": "MoviesFlatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "MoviesFlatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["MoviesEmbedding", 0, 0, {}]]], "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 18}}
Ï
	variables
 regularization_losses
!trainable_variables
"	keras_api
c__call__
*d&call_and_return_all_conditional_losses"À
_tf_keras_layer¦{"name": "UsersFlatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "UsersFlatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["UsersEmbedding", 0, 0, {}]]], "shared_object_id": 11, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 19}}
¸
#	variables
$regularization_losses
%trainable_variables
&	keras_api
e__call__
*f&call_and_return_all_conditional_losses"©
_tf_keras_layer{"name": "DotProduct", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dot", "config": {"name": "DotProduct", "trainable": true, "dtype": "float32", "axes": 1, "normalize": false}, "inbound_nodes": [[["MoviesFlatten", 0, 0, {}], ["UsersFlatten", 0, 0, {}]]], "shared_object_id": 12, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 19]}, {"class_name": "TensorShape", "items": [null, 19]}]}
I
'iter
	(decay
)learning_rate
*momentum"
	optimizer
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
Ê

+layers
,layer_regularization_losses
-metrics
.non_trainable_variables

	variables
/layer_metrics
regularization_losses
trainable_variables
Z__call__
[_default_save_signature
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
,
iserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
-:+	óK2MoviesEmbedding/embeddings
'
0"
trackable_list_wrapper
'
g0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
­

0layers
1layer_regularization_losses
2metrics
3non_trainable_variables
	variables
4layer_metrics
regularization_losses
trainable_variables
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
,:*	â2UsersEmbedding/embeddings
'
0"
trackable_list_wrapper
'
h0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
­

5layers
6layer_regularization_losses
7metrics
8non_trainable_variables
	variables
9layer_metrics
regularization_losses
trainable_variables
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­

:layers
;layer_regularization_losses
<metrics
=non_trainable_variables
	variables
>layer_metrics
regularization_losses
trainable_variables
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­

?layers
@layer_regularization_losses
Ametrics
Bnon_trainable_variables
	variables
Clayer_metrics
 regularization_losses
!trainable_variables
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­

Dlayers
Elayer_regularization_losses
Fmetrics
Gnon_trainable_variables
#	variables
Hlayer_metrics
$regularization_losses
%trainable_variables
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
:	 (2SGD/iter
: (2	SGD/decay
: (2SGD/learning_rate
: (2SGD/momentum
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
5
I0
J1
K2"
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
'
h0"
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
 "
trackable_dict_wrapper
Ô
	Ltotal
	Mcount
N	variables
O	keras_api"
_tf_keras_metric{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 20}

	Ptotal
	Qcount
R
_fn_kwargs
S	variables
T	keras_api"Å
_tf_keras_metricª{"class_name": "MeanMetricWrapper", "name": "mae", "dtype": "float32", "config": {"name": "mae", "dtype": "float32", "fn": "mean_absolute_error"}, "shared_object_id": 16}

	Utotal
	Vcount
W
_fn_kwargs
X	variables
Y	keras_api"Ä
_tf_keras_metric©{"class_name": "MeanMetricWrapper", "name": "mse", "dtype": "float32", "config": {"name": "mse", "dtype": "float32", "fn": "mean_squared_error"}, "shared_object_id": 17}
:  (2total
:  (2count
.
L0
M1"
trackable_list_wrapper
-
N	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
P0
Q1"
trackable_list_wrapper
-
S	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
U0
V1"
trackable_list_wrapper
-
X	variables"
_generic_user_object
Î2Ë
@__inference_MatrixFactorizationReccomender_layer_call_fn_1517886
@__inference_MatrixFactorizationReccomender_layer_call_fn_1518067
@__inference_MatrixFactorizationReccomender_layer_call_fn_1518077
@__inference_MatrixFactorizationReccomender_layer_call_fn_1517977À
·²³
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

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
2
"__inference__wrapped_model_1517787å
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *U¢R
PM
$!

UsersInputÿÿÿÿÿÿÿÿÿ
%"
MoviesInputÿÿÿÿÿÿÿÿÿ
º2·
[__inference_MatrixFactorizationReccomender_layer_call_and_return_conditional_losses_1518117
[__inference_MatrixFactorizationReccomender_layer_call_and_return_conditional_losses_1518157
[__inference_MatrixFactorizationReccomender_layer_call_and_return_conditional_losses_1518003
[__inference_MatrixFactorizationReccomender_layer_call_and_return_conditional_losses_1518029À
·²³
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

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Û2Ø
1__inference_MoviesEmbedding_layer_call_fn_1518170¢
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
ö2ó
L__inference_MoviesEmbedding_layer_call_and_return_conditional_losses_1518186¢
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
Ú2×
0__inference_UsersEmbedding_layer_call_fn_1518199¢
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
õ2ò
K__inference_UsersEmbedding_layer_call_and_return_conditional_losses_1518215¢
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
Ù2Ö
/__inference_MoviesFlatten_layer_call_fn_1518220¢
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
ô2ñ
J__inference_MoviesFlatten_layer_call_and_return_conditional_losses_1518226¢
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
Ø2Õ
.__inference_UsersFlatten_layer_call_fn_1518231¢
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
ó2ð
I__inference_UsersFlatten_layer_call_and_return_conditional_losses_1518237¢
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
Ö2Ó
,__inference_DotProduct_layer_call_fn_1518243¢
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
ñ2î
G__inference_DotProduct_layer_call_and_return_conditional_losses_1518255¢
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
´2±
__inference_loss_fn_0_1518266
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
´2±
__inference_loss_fn_1_1518277
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
ÚB×
%__inference_signature_wrapper_1518057MoviesInput
UsersInput"
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
 Ï
G__inference_DotProduct_layer_call_and_return_conditional_losses_1518255Z¢W
P¢M
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¦
,__inference_DotProduct_layer_call_fn_1518243vZ¢W
P¢M
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿô
[__inference_MatrixFactorizationReccomender_layer_call_and_return_conditional_losses_1518003g¢d
]¢Z
PM
$!

UsersInputÿÿÿÿÿÿÿÿÿ
%"
MoviesInputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ô
[__inference_MatrixFactorizationReccomender_layer_call_and_return_conditional_losses_1518029g¢d
]¢Z
PM
$!

UsersInputÿÿÿÿÿÿÿÿÿ
%"
MoviesInputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ï
[__inference_MatrixFactorizationReccomender_layer_call_and_return_conditional_losses_1518117b¢_
X¢U
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ï
[__inference_MatrixFactorizationReccomender_layer_call_and_return_conditional_losses_1518157b¢_
X¢U
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ì
@__inference_MatrixFactorizationReccomender_layer_call_fn_1517886g¢d
]¢Z
PM
$!

UsersInputÿÿÿÿÿÿÿÿÿ
%"
MoviesInputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿÌ
@__inference_MatrixFactorizationReccomender_layer_call_fn_1517977g¢d
]¢Z
PM
$!

UsersInputÿÿÿÿÿÿÿÿÿ
%"
MoviesInputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÇ
@__inference_MatrixFactorizationReccomender_layer_call_fn_1518067b¢_
X¢U
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿÇ
@__inference_MatrixFactorizationReccomender_layer_call_fn_1518077b¢_
X¢U
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ¯
L__inference_MoviesEmbedding_layer_call_and_return_conditional_losses_1518186_/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
1__inference_MoviesEmbedding_layer_call_fn_1518170R/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿª
J__inference_MoviesFlatten_layer_call_and_return_conditional_losses_1518226\3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
/__inference_MoviesFlatten_layer_call_fn_1518220O3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ®
K__inference_UsersEmbedding_layer_call_and_return_conditional_losses_1518215_/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
0__inference_UsersEmbedding_layer_call_fn_1518199R/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ©
I__inference_UsersFlatten_layer_call_and_return_conditional_losses_1518237\3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_UsersFlatten_layer_call_fn_1518231O3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÅ
"__inference__wrapped_model_1517787_¢\
U¢R
PM
$!

UsersInputÿÿÿÿÿÿÿÿÿ
%"
MoviesInputÿÿÿÿÿÿÿÿÿ
ª "7ª4
2

DotProduct$!

DotProductÿÿÿÿÿÿÿÿÿ<
__inference_loss_fn_0_1518266¢

¢ 
ª " <
__inference_loss_fn_1_1518277¢

¢ 
ª " à
%__inference_signature_wrapper_1518057¶w¢t
¢ 
mªj
4
MoviesInput%"
MoviesInputÿÿÿÿÿÿÿÿÿ
2

UsersInput$!

UsersInputÿÿÿÿÿÿÿÿÿ"7ª4
2

DotProduct$!

DotProductÿÿÿÿÿÿÿÿÿ