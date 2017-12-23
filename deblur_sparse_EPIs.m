function [] = deblur_sparse_EPIs(input_dir, output_dir, kernel_width)
lambda = 2e3;
alpha = 1/2;

blurred_EPIs = dir(fullfile(input_dir, '*.png'));

for i = 1:length(blurred_EPIs)
    y = double(imread(strcat(input_dir, blurred_EPIs(i).name)))./255.0;
    sigma = 1.5;
    kernel=fspecial('gaussian',[1 kernel_width], 1.5);
    deblurred_output = fast_deconv(y, kernel, lambda, alpha);
    filename = strcat(output_dir, blurred_EPIs(i).name);
    imwrite(deblurred_output, filename);
end
end