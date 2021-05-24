# Test Cascade 1
python3 testReal.py --cuda --dataRoot data/iiw-dataset/data --imList NYUTest.txt \
    --testRoot NYU_cascade1 --isLight --isBS --level 2 \
    --experiment0 models/check_cascadeNYU0 --nepoch0 2 \
    --experimentLight0 models/check_cascadeLight0_sg12_offset1.0 --nepochLight0 10 \
    --experimentBS0 models/checkBs_cascade0_w320_h240 \
    --experiment1 models/check_cascadeNYU1 --nepoch1 3 \
    --experimentLight1 models/check_cascadeLight1_sg12_offset1.0 --nepochLight1 10 \
    --experimentBS1 models/checkBs_cascade1_w320_h240 \
