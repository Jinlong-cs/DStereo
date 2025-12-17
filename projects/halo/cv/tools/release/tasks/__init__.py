from .detection import detection
from .eye_status import eye_status
from .eye_visibility import eye_visibility
from .face3d import face3d
from .face_attr import face_attr
from .face_multitask import face_multitask
from .face_quality import face_quality
from .faceid import faceid
from .gaze import gaze
from .gesture2d import gesture2d
from .hand3d import hand3d
from .human3d import human3d
from .landmark import landmark
from .pccr import pccr
from .pnpnet import pnpnet
from .pupil_seg import pupil_seg
from .smoke_phone import smoke_phone

__all__ = ["task_dependencies"]


task_dependencies = (
    detection
    + face_quality
    + eye_status
    + eye_visibility
    + face_attr
    + face_multitask
    + face3d
    + faceid
    + gaze
    + hand3d
    + human3d
    + landmark
    + pnpnet
    + pupil_seg
    + smoke_phone
    + gesture2d
    + pccr
)
