train
python main.py ctdet --arch mv2_10 --exp_id mv2_center_person_512 --batch_size 1 --master_batch 1 --lr 1.25e-4  --gpus 0 --fix_res --num_workers 4 --num_epochs 300 --dataset pascal --input_h 512 --input_w 512

onnx
python pytorch2onnx.py ctdet --arch mv2_10 --load_model ../exp/ctdet/mv2_center_person_512/model_last.pth --input_h 512 --input_w 512 --fix_res

simplify
python -m onnxsim PiggyPoint.onnx PiggyPoint_sim.onnx

ncnn
./onnx2ncnn PiggyPoint_sim.onnx PiggyPoint.param PiggyPoint.bin

test
python demo.py ctdet --demo ../test --debug 4 --arch mv2_10 --load_model ../exp/ctdet/mv2_center_person_512/model_last.pth --input_h 512 --input_w 512 --fix_res --num_classes 1