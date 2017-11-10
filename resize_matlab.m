horizontal_EPIs = dir(fullfile('EPI\', '*.png'));

for i = 1:length(horizontal_EPIs)
   
   EPI = imread(strcat('EPI/', horizontal_EPIs(i).name));
   [m,n,p] = size(EPI);
   
   % Gaussian sigma is a parameter defined by Wu et al. CVPR 2017
   g_x = fspecial('gaussian',[1 17], 1.5);
   
   EPI = imfilter(EPI, g_x);
   EPI_2 = EPI(1:2:end, :, :);
   downsampled2 = imresize(EPI_2, [m n]);
   imwrite(uint8(downsampled2), strcat('fixedblur_downsampled2/', horizontal_EPIs(i).name));
   
   % [2017-09-28] JT: Blur again to remove aliasing for 4x downsample
   EPI_2 = imfilter(EPI_2, g_x);
   EPI_4 = EPI_2(1:2:end, :, :);
   downsampled4 = imresize(EPI_4, [m n]);
   imwrite(uint8(downsampled4), strcat('fixedblur_downsampled4/', horizontal_EPIs(i).name));
   
   
   imwrite(uint8(EPI), strcat('fixedblur_groundtruthblurred/', horizontal_EPIs(i).name));
end