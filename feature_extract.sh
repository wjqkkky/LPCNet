# Place in 16k-LP7 from TSPSpeech.iso and run to concatenate wave files
# into one headerless training file
for i in mandarin_female/wavs/*.s16 
do
./dump_data -test $i mandarin_female/feature_extract/${i##*/}.f32
done
