cmake -DCMAKE_PREFIX_PATH="& {python3 -c `"import torch; print(torch.utils.cmake_prefix_path)`"}" `
	-DCMAKE_BUILD_TYPE=Release -S . -B bin && cmake --build bin --target ALL_BUILD -j4 && ./bin/AI.exe
