python prepare_data.py
python train.py --model_save_path "./my_model" --num_train_epoch 3
python pipeline.py --model_load_path "./my_model" --input_file "project_test_sentences.txt" --output_file "final_results.json"