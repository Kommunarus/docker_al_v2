import os
import json
import copy
import random


def write_json(list_files, m, path_to_out, full_train_json='/home/neptun/PycharmProjects/datasets/coco/instances_train2017.json'):

    current_label = 1  # cat

    with open(full_train_json) as f:
        razmetka = json.load(f)

    categories = razmetka['categories']
    annotations = razmetka['annotations']
    images = razmetka['images']
    info = razmetka['info']
    licenses = razmetka['licenses']

    new_image = []
    a = []
    for row in images:
        if row['file_name'] in list_files:
            a.append(row['id'])
            copy_row = copy.deepcopy(row)
            new_image.append(copy_row)

    new_annotation = []
    count_good_image = []

    for row in annotations:
        if row['category_id'] == current_label and row['image_id'] in a:
            copy_row = copy.deepcopy(row)
            copy_row['segmentation'] = []
            new_annotation.append(copy_row)
            count_good_image.append(row['image_id'])
    count_good_image = len(list(set(count_good_image)))

    b = []
    for row in new_annotation:
        b.append(row['image_id'])
    b = list(set(b))

    print('file {}, {} / {}'.format(m, len(b), len(list_files)))

    # c = list(set(list_files) - set(b))
    # c = random.sample(c, k=min(3*len(b), len(c)))
    new_image2 = []
    for row in images:
        if row['file_name'] in list_files:
        # if row['id'] in b:
            copy_row = copy.deepcopy(row)
            new_image2.append(copy_row)


    new_razmetka = dict(annotations=new_annotation, images=new_image2,
                        categories=categories, info=info, licenses=licenses)

    with open(os.path.join(path_to_out, f'{m}.json'), 'w') as f:
        f.write(json.dumps(new_razmetka))

    return count_good_image


if __name__ == '__main__':
    list_files1 = [
        "000000000395.jpg",
        "000000000531.jpg",
        "000000001431.jpg",
        "000000001912.jpg",
        "000000002232.jpg",
        "000000002295.jpg",
        "000000003366.jpg",
        "000000004069.jpg",
        "000000004376.jpg",
        "000000004537.jpg",
        "000000005701.jpg",
        "000000005994.jpg",
        "000000006151.jpg",
        "000000007256.jpg",
        "000000008179.jpg",
        "000000008196.jpg",
        "000000009885.jpg",
        "000000010058.jpg",
        "000000010728.jpg",
        "000000011931.jpg",
        "000000012057.jpg",
        "000000012824.jpg",
        "000000013383.jpg",
        "000000013670.jpg",
        "000000015757.jpg",
        "000000016233.jpg",
        "000000016240.jpg",
        "000000017401.jpg",
        "000000017450.jpg",
        "000000018555.jpg",
        "000000018886.jpg",
        "000000019134.jpg",
        "000000020711.jpg",
        "000000022757.jpg",
        "000000022775.jpg",
        "000000023451.jpg",
        "000000023729.jpg",
        "000000023906.jpg",
        "000000024100.jpg",
        "000000024674.jpg",
        "000000026430.jpg",
        "000000026598.jpg",
        "000000026762.jpg",
        "000000027109.jpg",
        "000000027252.jpg",
        "000000027329.jpg",
        "000000028738.jpg",
        "000000029524.jpg",
        "000000032578.jpg",
        "000000032667.jpg",
        "000000032689.jpg",
        "000000033111.jpg",
        "000000033405.jpg",
        "000000035643.jpg",
        "000000036508.jpg",
        "000000037175.jpg",
        "000000037671.jpg",
        "000000038732.jpg",
        "000000039312.jpg",
        "000000039509.jpg",
        "000000039828.jpg",
        "000000040055.jpg",
        "000000040158.jpg",
        "000000040210.jpg",
        "000000040995.jpg",
        "000000041678.jpg",
        "000000041763.jpg",
        "000000042042.jpg",
        "000000042516.jpg",
        "000000042690.jpg",
        "000000042740.jpg",
        "000000042975.jpg",
        "000000043353.jpg",
        "000000043367.jpg",
        "000000043734.jpg",
        "000000044922.jpg",
        "000000045941.jpg",
        "000000048103.jpg",
        "000000049065.jpg",
        "000000050277.jpg",
        "000000050322.jpg",
        "000000050945.jpg",
        "000000051012.jpg",
        "000000051680.jpg",
        "000000053754.jpg",
        "000000054065.jpg",
        "000000054295.jpg",
        "000000055085.jpg",
        "000000055288.jpg",
        "000000056092.jpg",
        "000000056506.jpg",
        "000000056616.jpg",
        "000000056978.jpg",
        "000000057460.jpg",
        "000000058008.jpg",
        "000000062716.jpg",
        "000000063409.jpg",
        "000000063566.jpg",
        "000000063835.jpg",
        "000000063848.jpg",
        "000000063950.jpg",
        "000000064196.jpg",
        "000000064896.jpg",
        "000000065557.jpg",
        "000000065773.jpg",
        "000000066628.jpg",
        "000000066734.jpg",
        "000000067116.jpg",
        "000000067674.jpg",
        "000000067995.jpg",
        "000000068335.jpg",
        "000000068996.jpg",
        "000000069383.jpg",
        "000000069532.jpg",
        "000000071214.jpg",
        "000000072961.jpg",
        "000000073503.jpg",
        "000000074794.jpg",
        "000000075162.jpg",
        "000000075427.jpg",
        "000000076203.jpg",
        "000000076619.jpg",
        "000000077355.jpg",
        "000000077806.jpg",
        "000000077847.jpg",
        "000000077936.jpg",
        "000000078593.jpg",
        "000000078782.jpg",
        "000000080208.jpg",
        "000000080713.jpg",
        "000000080749.jpg",
        "000000081166.jpg",
        "000000081842.jpg",
        "000000081995.jpg",
        "000000082705.jpg",
        "000000084540.jpg",
        "000000084735.jpg",
        "000000085092.jpg",
        "000000086738.jpg",
        "000000086989.jpg",
        "000000087377.jpg",
        "000000087738.jpg",
        "000000088034.jpg",
        "000000089369.jpg",
        "000000090349.jpg",
        "000000090830.jpg",
        "000000091334.jpg",
        "000000091566.jpg",
        "000000092526.jpg",
        "000000093025.jpg",
        "000000093506.jpg",
        "000000094563.jpg",
        "000000094793.jpg",
        "000000095050.jpg",
        "000000095486.jpg",
        "000000096679.jpg",
        "000000096762.jpg",
        "000000096828.jpg",
        "000000097314.jpg",
        "000000097865.jpg",
        "000000097939.jpg",
        "000000098495.jpg",
        "000000098631.jpg",
        "000000098903.jpg",
        "000000099030.jpg",
        "000000099135.jpg",
        "000000102469.jpg",
        "000000102610.jpg",
        "000000104393.jpg",
        "000000104725.jpg",
        "000000104811.jpg",
        "000000105030.jpg",
        "000000105582.jpg",
        "000000106210.jpg",
        "000000106748.jpg",
        "000000106909.jpg",
        "000000108620.jpg",
        "000000109776.jpg",
        "000000110259.jpg",
        "000000111492.jpg",
        "000000111845.jpg",
        "000000111998.jpg",
        "000000112478.jpg",
        "000000114034.jpg",
        "000000114473.jpg",
        "000000115014.jpg",
        "000000115080.jpg",
        "000000116526.jpg",
        "000000116824.jpg",
        "000000118785.jpg",
        "000000119081.jpg",
        "000000119171.jpg",
        "000000119494.jpg",
        "000000119693.jpg",
        "000000119979.jpg",
        "000000120535.jpg",
        "000000121994.jpg",
        "000000122421.jpg",
        "000000126054.jpg",
        "000000126090.jpg",
        "000000126163.jpg",
        "000000126182.jpg",
        "000000126536.jpg",
        "000000127899.jpg",
        "000000127920.jpg",
        "000000128311.jpg",
        "000000128691.jpg",
        "000000129117.jpg",
        "000000130132.jpg",
        "000000130181.jpg",
        "000000132476.jpg",
        "000000133503.jpg",
        "000000134012.jpg",
        "000000134198.jpg",
        "000000135029.jpg",
        "000000135158.jpg",
        "000000135554.jpg",
        "000000136373.jpg",
        "000000137212.jpg",
        "000000138937.jpg",
        "000000139568.jpg",
        "000000139728.jpg",
        "000000140539.jpg",
        "000000141586.jpg",
        "000000141955.jpg",
        "000000142274.jpg",
        "000000142953.jpg",
        "000000144049.jpg",
        "000000144438.jpg",
        "000000144534.jpg",
        "000000145238.jpg",
        "000000145266.jpg",
        "000000145429.jpg",
        "000000145638.jpg",
        "000000145815.jpg",
        "000000146084.jpg",
        "000000146256.jpg",
        "000000147488.jpg",
        "000000147506.jpg",
        "000000147787.jpg",
        "000000148301.jpg",
        "000000148422.jpg",
        "000000149835.jpg",
        "000000149884.jpg",
        "000000149896.jpg",
        "000000152096.jpg",
        "000000152702.jpg",
        "000000152913.jpg",
        "000000152962.jpg",
        "000000152965.jpg",
        "000000153692.jpg",
        "000000154167.jpg",
        "000000154869.jpg",
        "000000157041.jpg",
        "000000158601.jpg",
        "000000158897.jpg",
        "000000161686.jpg",
        "000000162753.jpg",
        "000000163782.jpg",
        "000000165056.jpg",
        "000000165341.jpg",
        "000000166207.jpg",
        "000000166340.jpg",
        "000000168505.jpg",
        "000000168843.jpg",
        "000000169584.jpg",
        "000000169947.jpg",
        "000000171255.jpg",
        "000000172036.jpg",
        "000000173425.jpg",
        "000000175479.jpg",
        "000000175969.jpg",
        "000000177486.jpg",
        "000000177516.jpg",
        "000000178810.jpg",
        "000000179876.jpg",
        "000000180362.jpg",
        "000000180818.jpg",
        "000000181980.jpg",
        "000000182903.jpg",
        "000000182933.jpg",
        "000000183268.jpg",
        "000000185432.jpg",
        "000000187262.jpg",
        "000000187286.jpg",
        "000000187371.jpg",
        "000000187496.jpg",
        "000000187738.jpg",
        "000000188631.jpg",
        "000000188987.jpg",
        "000000189127.jpg",
        "000000190580.jpg",
        "000000190718.jpg",
        "000000191262.jpg",
        "000000191382.jpg",
        "000000191740.jpg",
        "000000191854.jpg",
        "000000192077.jpg",
        "000000192843.jpg",
        "000000193328.jpg",
        "000000193387.jpg",
        "000000193481.jpg",
        "000000195716.jpg",
        "000000195816.jpg",
        "000000196104.jpg",
        "000000197155.jpg",
        "000000197570.jpg",
        "000000198396.jpg",
        "000000198493.jpg",
        "000000199626.jpg",
        "000000199815.jpg",
        "000000200882.jpg",
        "000000203879.jpg",
        "000000204589.jpg",
        "000000204940.jpg",
        "000000205940.jpg",
        "000000206398.jpg",
        "000000206400.jpg",
        "000000206542.jpg",
        "000000206548.jpg",
        "000000206670.jpg",
        "000000207437.jpg",
        "000000208140.jpg",
        "000000208318.jpg",
        "000000208459.jpg",
        "000000208649.jpg",
        "000000208945.jpg",
        "000000209957.jpg",
        "000000211402.jpg",
        "000000211863.jpg",
        "000000212203.jpg",
        "000000212363.jpg",
        "000000212647.jpg",
        "000000213141.jpg",
        "000000213305.jpg",
        "000000213578.jpg",
        "000000214800.jpg",
        "000000215353.jpg",
        "000000215424.jpg",
        "000000218939.jpg",
        "000000218985.jpg",
        "000000219063.jpg",
        "000000219410.jpg",
        "000000220037.jpg",
        "000000220306.jpg",
        "000000220981.jpg",
        "000000221380.jpg",
        "000000221560.jpg",
        "000000221684.jpg",
        "000000221915.jpg",
        "000000222086.jpg",
        "000000222681.jpg",
        "000000223317.jpg",
        "000000223369.jpg",
        "000000223404.jpg",
        "000000223660.jpg",
        "000000224594.jpg",
        "000000228376.jpg",
        "000000229491.jpg",
        "000000230490.jpg",
        "000000230516.jpg",
        "000000231295.jpg",
        "000000231958.jpg",
        "000000232275.jpg",
        "000000232524.jpg",
        "000000233337.jpg",
        "000000233539.jpg",
        "000000233808.jpg",
        "000000234296.jpg",
        "000000234343.jpg",
        "000000234522.jpg",
        "000000234734.jpg",
        "000000236290.jpg",
        "000000236338.jpg",
        "000000236516.jpg",
        "000000238607.jpg",
        "000000239396.jpg",
        "000000240287.jpg",
        "000000240344.jpg",
        "000000241355.jpg",
        "000000241557.jpg",
        "000000242467.jpg",
        "000000242592.jpg",
        "000000242762.jpg",
        "000000243171.jpg",
        "000000243875.jpg",
        "000000243947.jpg",
        "000000244003.jpg",
        "000000244050.jpg",
        "000000244065.jpg",
        "000000244088.jpg",
        "000000244737.jpg",
        "000000244975.jpg",
        "000000245733.jpg",
        "000000248206.jpg",
        "000000248403.jpg",
        "000000249151.jpg",
        "000000249964.jpg",
        "000000250365.jpg",
        "000000250543.jpg",
        "000000250630.jpg",
        "000000250706.jpg",
        "000000250777.jpg",
        "000000251246.jpg",
        "000000251608.jpg",
        "000000251741.jpg",
        "000000251751.jpg",
        "000000252364.jpg",
        "000000252439.jpg",
        "000000253266.jpg",
        "000000253576.jpg",
        "000000253718.jpg",
        "000000254392.jpg",
        "000000254568.jpg",
        "000000254822.jpg",
        "000000255181.jpg",
        "000000255470.jpg",
        "000000256184.jpg",
        "000000256637.jpg",
        "000000257864.jpg",
        "000000258499.jpg",
        "000000258555.jpg",
        "000000259336.jpg",
        "000000260738.jpg",
        "000000261225.jpg",
        "000000261761.jpg",
        "000000262146.jpg",
        "000000263084.jpg",
        "000000263311.jpg",
        "000000263574.jpg",
        "000000263881.jpg",
        "000000264737.jpg",
        "000000264805.jpg",
        "000000265743.jpg",
        "000000266099.jpg",
        "000000266371.jpg",
        "000000266563.jpg",
        "000000267290.jpg",
        "000000267560.jpg",
        "000000267780.jpg",
        "000000267840.jpg",
        "000000268192.jpg",
        "000000270112.jpg",
        "000000270818.jpg",
        "000000271057.jpg",
        "000000272058.jpg",
        "000000272162.jpg",
        "000000272250.jpg",
        "000000272599.jpg",
        "000000273369.jpg",
        "000000273814.jpg",
        "000000274105.jpg",
        "000000274109.jpg",
        "000000274270.jpg",
        "000000274657.jpg",
        "000000274949.jpg",
        "000000275581.jpg",
        "000000276515.jpg",
        "000000276852.jpg",
        "000000277050.jpg",
        "000000278175.jpg",
        "000000278653.jpg",
        "000000279407.jpg",
        "000000279428.jpg",
        "000000281074.jpg",
        "000000281615.jpg",
        "000000281649.jpg",
        "000000282155.jpg",
        "000000282553.jpg",
        "000000282952.jpg",
        "000000283589.jpg",
        "000000283678.jpg",
        "000000283753.jpg",
        "000000284651.jpg",
        "000000284851.jpg",
        "000000285355.jpg",
        "000000285978.jpg",
        "000000286469.jpg",
        "000000286770.jpg",
        "000000286874.jpg",
        "000000287656.jpg",
        "000000288547.jpg",
        "000000289621.jpg",
        "000000290289.jpg",
        "000000290678.jpg",
        "000000291498.jpg",
        "000000291921.jpg",
        "000000293385.jpg",
        "000000293756.jpg",
        "000000294223.jpg",
        "000000294615.jpg",
        "000000294718.jpg",
        "000000295329.jpg",
        "000000295695.jpg",
        "000000296002.jpg",
        "000000296038.jpg",
        "000000296289.jpg",
        "000000297877.jpg",
        "000000298914.jpg",
        "000000300023.jpg",
        "000000301347.jpg",
        "000000301708.jpg",
        "000000302655.jpg",
        "000000303250.jpg",
        "000000306902.jpg",
        "000000306967.jpg",
        "000000307968.jpg",
        "000000308504.jpg",
        "000000308576.jpg",
        "000000309993.jpg",
        "000000310206.jpg",
        "000000310558.jpg",
        "000000311004.jpg",
        "000000311087.jpg",
        "000000311104.jpg",
        "000000311669.jpg",
        "000000312744.jpg",
        "000000312826.jpg",
        "000000313334.jpg",
        "000000313812.jpg",
        "000000313983.jpg",
        "000000314139.jpg",
        "000000314189.jpg",
        "000000314899.jpg",
        "000000315073.jpg",
        "000000317102.jpg",
        "000000318616.jpg",
        "000000318857.jpg",
        "000000319706.jpg",
        "000000319866.jpg",
        "000000319908.jpg",
        "000000320386.jpg",
        "000000320643.jpg",
        "000000321070.jpg",
        "000000321476.jpg",
        "000000321903.jpg",
        "000000322500.jpg",
        "000000322945.jpg",
        "000000323164.jpg",
        "000000323585.jpg",
        "000000323979.jpg",
        "000000324000.jpg",
        "000000324026.jpg",
        "000000324143.jpg",
        "000000324823.jpg",
        "000000324969.jpg",
        "000000325783.jpg",
        "000000326048.jpg",
        "000000326666.jpg",
        "000000327221.jpg",
        "000000327843.jpg",
        "000000330186.jpg",
        "000000330570.jpg",
        "000000331883.jpg",
        "000000332777.jpg",
        "000000333613.jpg",
        "000000333876.jpg",
        "000000333916.jpg",
        "000000334645.jpg",
        "000000334732.jpg",
        "000000335624.jpg",
        "000000336426.jpg",
        "000000336682.jpg",
        "000000337953.jpg",
        "000000340102.jpg",
        "000000340226.jpg",
        "000000340263.jpg",
        "000000340270.jpg",
        "000000340610.jpg",
        "000000340962.jpg",
        "000000340988.jpg",
        "000000341429.jpg",
        "000000341623.jpg",
        "000000342817.jpg",
        "000000343035.jpg",
        "000000343057.jpg",
        "000000343140.jpg",
        "000000343407.jpg",
        "000000343458.jpg",
        "000000343878.jpg",
        "000000344222.jpg",
        "000000344675.jpg",
        "000000345691.jpg",
        "000000345987.jpg",
        "000000346366.jpg",
        "000000347354.jpg",
        "000000347483.jpg",
        "000000347507.jpg",
        "000000347848.jpg",
        "000000348186.jpg",
        "000000348235.jpg",
        "000000348609.jpg",
        "000000349663.jpg",
        "000000351154.jpg",
        "000000351528.jpg",
        "000000351534.jpg",
        "000000352125.jpg",
        "000000352217.jpg",
        "000000352234.jpg",
        "000000352755.jpg",
        "000000353124.jpg",
        "000000353408.jpg",
        "000000354165.jpg",
        "000000354237.jpg",
        "000000357312.jpg",
        "000000357356.jpg",
        "000000357424.jpg",
        "000000357925.jpg",
        "000000358307.jpg",
        "000000358586.jpg",
        "000000359232.jpg",
        "000000359789.jpg",
        "000000360068.jpg",
        "000000360896.jpg",
        "000000361451.jpg",
        "000000361819.jpg",
        "000000361924.jpg",
        "000000361972.jpg",
        "000000362654.jpg",
        "000000363386.jpg",
        "000000363514.jpg",
        "000000364559.jpg",
        "000000364608.jpg",
        "000000364617.jpg",
        "000000364953.jpg",
        "000000365819.jpg",
        "000000366599.jpg",
        "000000366787.jpg",
        "000000367753.jpg",
        "000000368148.jpg",
        "000000368876.jpg",
        "000000369279.jpg",
        "000000369338.jpg",
        "000000369782.jpg",
        "000000370325.jpg",
        "000000370669.jpg",
        "000000370741.jpg",
        "000000370749.jpg",
        "000000372067.jpg",
        "000000372412.jpg",
        "000000372430.jpg",
        "000000372580.jpg",
        "000000373500.jpg",
        "000000374451.jpg",
        "000000375226.jpg",
        "000000375369.jpg",
        "000000375782.jpg",
        "000000376684.jpg",
        "000000378049.jpg",
        "000000378214.jpg",
        "000000378588.jpg",
        "000000378611.jpg",
        "000000378764.jpg",
        "000000378867.jpg",
        "000000378950.jpg",
        "000000379037.jpg",
        "000000380011.jpg",
        "000000380429.jpg",
        "000000381366.jpg",
        "000000381789.jpg",
        "000000381808.jpg",
        "000000382310.jpg",
        "000000382345.jpg",
        "000000382953.jpg",
        "000000383404.jpg",
        "000000383419.jpg",
        "000000383640.jpg",
        "000000385863.jpg",
        "000000386581.jpg",
        "000000386936.jpg",
        "000000386964.jpg",
        "000000387007.jpg",
        "000000387223.jpg",
        "000000387518.jpg",
        "000000387750.jpg",
        "000000388235.jpg",
        "000000388812.jpg",
        "000000389400.jpg",
        "000000389463.jpg",
        "000000389563.jpg",
        "000000389692.jpg",
        "000000389759.jpg",
        "000000389788.jpg",
        "000000391842.jpg",
        "000000392476.jpg",
        "000000392556.jpg",
        "000000392631.jpg",
        "000000394133.jpg",
        "000000396030.jpg",
        "000000397842.jpg",
        "000000397938.jpg",
        "000000398519.jpg",
        "000000399012.jpg",
        "000000399476.jpg",
        "000000399490.jpg",
        "000000399554.jpg",
        "000000399558.jpg",
        "000000399605.jpg",
        "000000399626.jpg",
        "000000399920.jpg",
        "000000401550.jpg",
        "000000403078.jpg",
        "000000403307.jpg",
        "000000403571.jpg",
        "000000404066.jpg",
        "000000404754.jpg",
        "000000405215.jpg",
        "000000405216.jpg",
        "000000405709.jpg",
        "000000406062.jpg",
        "000000407130.jpg",
        "000000410023.jpg",
        "000000410482.jpg",
        "000000411685.jpg",
        "000000412151.jpg",
        "000000412641.jpg",
        "000000413312.jpg",
        "000000413791.jpg",
        "000000414046.jpg",
        "000000415001.jpg",
        "000000415015.jpg",
        "000000415026.jpg",
        "000000415523.jpg",
        "000000416651.jpg",
        "000000416840.jpg",
        "000000417616.jpg",
        "000000418352.jpg",
        "000000419445.jpg",
        "000000419735.jpg",
        "000000419777.jpg",
        "000000421367.jpg",
        "000000422758.jpg",
        "000000422778.jpg",
        "000000425254.jpg",
        "000000425263.jpg",
        "000000426897.jpg",
        "000000428812.jpg",
        "000000429042.jpg",
        "000000429460.jpg",
        "000000432486.jpg",
        "000000432624.jpg",
        "000000433505.jpg",
        "000000433554.jpg",
        "000000433921.jpg",
        "000000434141.jpg",
        "000000434264.jpg",
        "000000435011.jpg",
        "000000435342.jpg",
        "000000435975.jpg",
        "000000436407.jpg",
        "000000437951.jpg",
        "000000438721.jpg",
        "000000439072.jpg",
        "000000439299.jpg",
        "000000439712.jpg",
        "000000439756.jpg",
        "000000439859.jpg",
        "000000440147.jpg",
        "000000440991.jpg",
        "000000441736.jpg",
        "000000442148.jpg",
        "000000442338.jpg",
        "000000442457.jpg",
        "000000442646.jpg",
        "000000442672.jpg",
        "000000442901.jpg",
        "000000444498.jpg",
        "000000444871.jpg",
        "000000445628.jpg",
        "000000446626.jpg",
        "000000446899.jpg",
        "000000447663.jpg",
        "000000447902.jpg",
        "000000448511.jpg",
        "000000449467.jpg",
        "000000449598.jpg",
        "000000450047.jpg",
        "000000450247.jpg",
        "000000450653.jpg",
        "000000451539.jpg",
        "000000452841.jpg",
        "000000452878.jpg",
        "000000453016.jpg",
        "000000454258.jpg",
        "000000455004.jpg",
        "000000456462.jpg",
        "000000457288.jpg",
        "000000457335.jpg",
        "000000457514.jpg",
        "000000458119.jpg",
        "000000458374.jpg",
        "000000459706.jpg",
        "000000460040.jpg",
        "000000460148.jpg",
        "000000462129.jpg",
        "000000462767.jpg",
        "000000463266.jpg",
        "000000463883.jpg",
        "000000465050.jpg",
        "000000465220.jpg",
        "000000465359.jpg",
        "000000466771.jpg",
        "000000467204.jpg",
        "000000467232.jpg",
        "000000468321.jpg",
        "000000468993.jpg",
        "000000469317.jpg",
        "000000469755.jpg",
        "000000470070.jpg",
        "000000470551.jpg",
        "000000470718.jpg",
        "000000470832.jpg",
        "000000470951.jpg",
        "000000471345.jpg",
        "000000471654.jpg",
        "000000471916.jpg",
        "000000472582.jpg",
        "000000473337.jpg",
        "000000474955.jpg",
        "000000475130.jpg",
        "000000475550.jpg",
        "000000476735.jpg",
        "000000476785.jpg",
        "000000477336.jpg",
        "000000477497.jpg",
        "000000477664.jpg",
        "000000478320.jpg",
        "000000478374.jpg",
        "000000480486.jpg",
        "000000481974.jpg",
        "000000482191.jpg",
        "000000482730.jpg",
        "000000482748.jpg",
        "000000483587.jpg",
        "000000483833.jpg",
        "000000484113.jpg",
        "000000484570.jpg",
        "000000485173.jpg",
        "000000486586.jpg",
        "000000486805.jpg",
        "000000487279.jpg",
        "000000487774.jpg",
        "000000488387.jpg",
        "000000490118.jpg",
        "000000490572.jpg",
        "000000490857.jpg",
        "000000491831.jpg",
        "000000492466.jpg",
        "000000493438.jpg",
        "000000494090.jpg",
        "000000494211.jpg",
        "000000494217.jpg",
        "000000494578.jpg",
        "000000494618.jpg",
        "000000494707.jpg",
        "000000495048.jpg",
        "000000495235.jpg",
        "000000495599.jpg",
        "000000495608.jpg",
        "000000495884.jpg",
        "000000496267.jpg",
        "000000497117.jpg",
        "000000497226.jpg",
        "000000497815.jpg",
        "000000498856.jpg",
        "000000499486.jpg",
        "000000499985.jpg",
        "000000500579.jpg",
        "000000500780.jpg",
        "000000501118.jpg",
        "000000502410.jpg",
        "000000502798.jpg",
        "000000502971.jpg",
        "000000503595.jpg",
        "000000504034.jpg",
        "000000504167.jpg",
        "000000504194.jpg",
        "000000504657.jpg",
        "000000506964.jpg",
        "000000507287.jpg",
        "000000509116.jpg",
        "000000509590.jpg",
        "000000510349.jpg",
        "000000510636.jpg",
        "000000510681.jpg",
        "000000512292.jpg",
        "000000512442.jpg",
        "000000513056.jpg",
        "000000516458.jpg",
        "000000516726.jpg",
        "000000516782.jpg",
        "000000516823.jpg",
        "000000517249.jpg",
        "000000517764.jpg",
        "000000518298.jpg",
        "000000518575.jpg",
        "000000520401.jpg",
        "000000520812.jpg",
        "000000521800.jpg",
        "000000522567.jpg",
        "000000522933.jpg",
        "000000523594.jpg",
        "000000525698.jpg",
        "000000527580.jpg",
        "000000528046.jpg",
        "000000528851.jpg",
        "000000529602.jpg",
        "000000531033.jpg",
        "000000531812.jpg",
        "000000532181.jpg",
        "000000532947.jpg",
        "000000533045.jpg",
        "000000533517.jpg",
        "000000534735.jpg",
        "000000534854.jpg",
        "000000536078.jpg",
        "000000537349.jpg",
        "000000538308.jpg",
        "000000538832.jpg",
        "000000538965.jpg",
        "000000540098.jpg",
        "000000541556.jpg",
        "000000541706.jpg",
        "000000541961.jpg",
        "000000542154.jpg",
        "000000543210.jpg",
        "000000544260.jpg",
        "000000544502.jpg",
        "000000544606.jpg",
        "000000544626.jpg",
        "000000545253.jpg",
        "000000545493.jpg",
        "000000545583.jpg",
        "000000546292.jpg",
        "000000546693.jpg",
        "000000547055.jpg",
        "000000547246.jpg",
        "000000548979.jpg",
        "000000549597.jpg",
        "000000550478.jpg",
        "000000550745.jpg",
        "000000550842.jpg",
        "000000551082.jpg",
        "000000551677.jpg",
        "000000551908.jpg",
        "000000552420.jpg",
        "000000553000.jpg",
        "000000553108.jpg",
        "000000553549.jpg",
        "000000553859.jpg",
        "000000554367.jpg",
        "000000554464.jpg",
        "000000555640.jpg",
        "000000555800.jpg",
        "000000557709.jpg",
        "000000558089.jpg",
        "000000558668.jpg",
        "000000558690.jpg",
        "000000559086.jpg",
        "000000559511.jpg",
        "000000561082.jpg",
        "000000561424.jpg",
        "000000561613.jpg",
        "000000562015.jpg",
        "000000562398.jpg",
        "000000563477.jpg",
        "000000564270.jpg",
        "000000564289.jpg",
        "000000564596.jpg",
        "000000565870.jpg",
        "000000566274.jpg",
        "000000566607.jpg",
        "000000566704.jpg",
        "000000567276.jpg",
        "000000567438.jpg",
        "000000568854.jpg",
        "000000569249.jpg",
        "000000570518.jpg",
        "000000570902.jpg",
        "000000570963.jpg",
        "000000570981.jpg",
        "000000571215.jpg",
        "000000571746.jpg",
        "000000571750.jpg",
        "000000572724.jpg",
        "000000572761.jpg",
        "000000573455.jpg",
        "000000574000.jpg",
        "000000574619.jpg",
        "000000576550.jpg",
        "000000577795.jpg",
        "000000578119.jpg",
        "000000579201.jpg",
        "000000579299.jpg",
        "000000579716.jpg",
        "000000580706.jpg",
        "000000581057.jpg",
        "000000581198.jpg",
        "000000581278.jpg",
        "000000581302.jpg"
    ]
    write_json(list_files1, 1, '/home/neptun/PycharmProjects/datasets/coco/labelstrain')
