clear all
close all
img_gray = zeros(600,600,5);
corrupted = zeros(600,600,5);
size = [600,600];

%Resize img and convert to grayscale
img1 = imread('original\test1.jpg');
img_gray(:,:,1) = im2double(rgb2gray(imresize(img1,size)));
img2 = imread('original\test2.jpg');
img_gray(:,:,2) = im2double(rgb2gray(imresize(img2,size)));
img3 = imread('original\test3.jpg');
img_gray(:,:,3) = im2double(rgb2gray(imresize(img3,size)));
img4 = imread('original\test4.jpg');
img_gray(:,:,4) = im2double(rgb2gray(imresize(img4,size)));
img5 = imread('original\test5.jpg');
img_gray(:,:,5) = im2double(rgb2gray(imresize(img5,size)));

% Generate binary masks and apply to test image
mask = full(logical(sprand(600,600,0.9)));
addr_gray = 'gray\';
addr_corrupted = 'corrupted\';

for i = 1:5
    %generate corrupted img
    corrupted(:,:,i) = img_gray(:,:,i) .* mask;
    %output img
    file_gray = fullfile(addr_gray, sprintf('gray_%d.jpg',i));
    imwrite(img_gray(:,:,i), file_gray);
    file_corrupted = fullfile(addr_corrupted, sprintf('corrupted_%d.jpg',i));
    imwrite(corrupted(:,:,i), file_corrupted);
end
save('mask.mat','mask');

%Mask guessing algorithm
guessMask = ones(600,600,2);
finalMask = ones(600,600);
corrupted_img = corrupted(:,:,5);
original_img = img_gray(:,:,5);

for x = 1:600
    for y = 1:600
        if ((corrupted(x,y,1) == 0) & (corrupted(x,y,2) == 0))
            guessMask(x,y,1) = 0;
        else 
            guessMask(x,y,1) = 1;
        end
    end
end

for x = 1:600
    for y = 1:600
        %Critical case
        if ((x == 1) | (x == 600) | (y == 1) | (y == 600))
            %Bot-left case
            if ((x == 1) & (y == 1))
                if ((corrupted_img(x,y) | corrupted_img(x+1,y) | corrupted_img(x,y+1) | corrupted_img(x+1,y+1)) == 0)
                    guessMask(x,y,2) = 1;
                else
                    guessMask(x,y,2) = 0;
                end
                %Top-left case
            elseif ((x == 1) & (y == 600))
                if ((corrupted_img(x,y) | corrupted_img(x+1,y) |  corrupted_img(x,y-1) |  corrupted_img(x+1,y-1)) == 0)
                    guessMask(x,y,2) = 1;
                else
                    guessMask(x,y,2) = 0;
                end
                %Bot-right case
            elseif ((x == 600) & (y == 1))
                if ((corrupted_img(x,y) | corrupted_img(x-1,y) | corrupted_img(x,y+1) | corrupted_img(x-1,y+1)) == 0)
                    guessMask(x,y,2) = 1;
                else
                    guessMask(x,y,2) = 0;
                end
                %Top-right case
            elseif ((x == 600) & (y == 600))
                if ((corrupted_img(x,y) | corrupted_img(x-1,y) | corrupted_img(x,y-1) | corrupted_img(x-1,y-1)) == 0)
                    guessMask(x,y,2) = 1;
                else
                    guessMask(x,y,2) = 0;
                end
                %left-edge
            elseif (x == 1)
                if ((corrupted_img(x,y) | corrupted_img(x+1,y) | corrupted_img(x,y+1) | corrupted_img(x,y-1) | corrupted_img(x+1,y-1) | corrupted_img(x+1,y+1)) == 0)
                    guessMask(x,y,2) = 1;
                else
                    guessMask(x,y,2) = 0;
                end
                %right edge
            elseif (x == 600)
                if ((corrupted_img(x,y) | corrupted_img(x-1,y) | corrupted_img(x,y+1) | corrupted_img(x,y-1) | corrupted_img(x-1,y-1) | corrupted_img(x-1,y+1)) == 0)
                    guessMask(x,y,2) = 1;
                else
                    guessMask(x,y,2) = 0;
                end
                %top edge
            elseif (y == 600)
                if ((corrupted_img(x,y) | corrupted_img(x+1,y) | corrupted_img(x-1,y) | corrupted_img(x,y-1) | corrupted_img(x-1,y-1) | corrupted_img(x+1,y-1)) == 0)
                    guessMask(x,y,2) = 1;
                else
                    guessMask(x,y,2) = 0;
                end
                %bot edge
            elseif (y == 1)
                if ((corrupted_img(x,y) | corrupted_img(x+1,y) | corrupted_img(x-1,y) | corrupted_img(x,y+1) | corrupted_img(x-1,y+1) | corrupted_img(x+1,y+1)) == 0)
                    guessMask(x,y,2) = 1;
                else
                    guessMask(x,y,2) = 0;
                end
            end
                 
            %Normal case
        else          
                if ((corrupted_img(x,y) | corrupted_img(x+1,y) | corrupted_img(x-1,y) | corrupted_img(x,y+1) | corrupted_img(x,y-1) | corrupted_img(x-1,y-1) | corrupted_img(x-1,y+1) | corrupted_img(x+1,y-1) | corrupted_img(x+1,y+1)) == 0)
                    guessMask(x,y,2) = 1;
                else
                    guessMask(x,y,2) = 0;
                end             
        end
    end
end

for x = 1:600
    for y = 1:600
        if (guessMask(x,y,1) == 0)
            if (guessMask(x,y,2) == 1)
                guessMask(x,y,1) = 1;
            end
        end
    end
end

%Calculating accuracy bewtwwen mask and guess
correctGuess = 0;
wrongGuess = 0;
successRate = 0;
for x = 1:600
    for y = 1:600
        if (guessMask(x,y,1) == mask(x,y))
            correctGuess = correctGuess + 1;
        else 
            wrongGuess = wrongGuess + 1;
        end
    end
end
successRate = correctGuess/(correctGuess + wrongGuess);


%IST algorithm
y = corrupted_img;
A = guessMask(:,:,1);
tol = 0.01;
MaxIter = 1000;
AA = @(x) x .* A;
AAdj = @(x) AA(x);
xinit = zeros(size);
xinit(:,:) = 0.5;
N0 = 4;
PPsi = @(x) wavedec2(x,N0,'db4');
[~,cbook] = wavedec2(xinit,N0,'db4');
PsiAdj = @(x) waverec2(x,cbook,'db4');
imshow(y);
title("Corrupted")

for lambda = [0.1]
    xstar = IST(y, lambda, PPsi, PsiAdj, AA, AAdj, MaxIter, tol, xinit);
    figure;
    imshow(xstar);
    title("Recovered for lambda = " + lambda)
end

%Difference of reconstructed and original
imshow(original_img - xstar);
title("Difference between original and recovered img")

%Calculate error (l1-norm)
error = 0;
for x = 1:600
    for y = 1:600
        error = error + (original_img(x,y) - xstar(x,y));
    end
end


function xx = ST(z, lambda)
xx = sign(z).*max(abs(z)-lambda,0);
end

function xstar = IST(y, lambda, Psi, PsiAdj, A, AAdj, MaxIter, tol, xinit)
eta = 0.1;

%replace variable
s = Psi(xinit);
B = @(x) A(PsiAdj(x));
BAdj = @(x) Psi(A(x));
g = @(x) norm(y-A(x))^2;

%Iteration
for k = 1: MaxIter
    if norm(y - B(s))/norm(y) >= tol
        s = ST(s - eta*BAdj(B(s) - y),eta*lambda);
    end
end
xstar = PsiAdj(s);
end