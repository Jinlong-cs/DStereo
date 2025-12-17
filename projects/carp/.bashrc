CARP_ROOT=$(dirname $(readlink -f $BASH_SOURCE))
HAT_ROOT=$(readlink -f $CARP_ROOT/../../)
WENET_ROOT=$(readlink -f $CARP_ROOT/plugins/wenet/)
export PYTHONPATH=$HAT_ROOT:$WENET_ROOT:$CARP_ROOT:$PYTHONPATH
export GLOG_minloglevel=1
