#! /bin/bash


subjects=(''subject10'' ''subject11'' ''subject13'' ''subject2'' ''subject4'' ''subject6'' ''subject8'' ''subject1''  ''subject12'' ''subject14'' ''subject3'' ''subject5'' ''subject7'' ''subject9'' ''subject15'' ''subject16'' ''subject17'' ''subject18'' ''subject19'' ''subject20'')

for subject in ${subjects[@]}

do
  echo "start pipeline for $subject"
  python ./get_radarfeatures.py -subj $subject
  wait
done

echo "finish pipeline"