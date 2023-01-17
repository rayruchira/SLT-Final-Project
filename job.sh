# Begin Job

universe=vanilla
requirements= TARGET.GPUSlot && CUDAGlobalMemoryMb >= 6144
request_GPUs=1
+GPUJob = true && NumJobStarts == 0

getenv = true

Initialdir= /u/ruchira/utcs-util
Executable = /lusr/bin/bash


+Group= "GRAD"
+Project= "AI_ROBOTICS"
+ProjectDescription = "Audio_Classification_Material"

# Begin final job information
Error= /u/ruchira/utcs-util/
Output= /u/ruchira/utcs-util/
Arguments = run.sh
Queue

# end of exp

