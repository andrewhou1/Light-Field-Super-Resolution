lambda = 2e3;
alpha = 1/2;

deblurred_EPIs = dir(fullfile('output_newloss\output_newloss\', '*.png'));

for i = 1:length(deblurred_EPIs)
   y = double(imread(strcat('output_newloss\output_newloss\', deblurred_EPIs(i).name)))./255.0;
   sigma = 1.5;
   kernel=fspecial('gaussian',[1 17], 1.5);
   deblurred_output = fast_deconv(y, kernel, lambda, alpha); 
   filename = strcat('deblurred_EPIs_fixed_lines/', deblurred_EPIs(i).name);
   imwrite(deblurred_output, filename);
end