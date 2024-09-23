function [ACC, nmi, Purity, Fscore, Precision, Recall, AR, Entropy]=clustering(S, cls_num, gt)
[C] = SpectralClustering(S,cls_num);

[ACC, nmi, Purity, Fscore, Precision, Recall, AR, Entropy] = Clustering8Measure(gt, C);
end
