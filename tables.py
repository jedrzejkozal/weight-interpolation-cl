import pathlib
import numpy as np
import mlflow
import tabulate


def main():
    standard_benchmarks()
    varing_n_tasks()
    interpolation_coef()


def standard_benchmarks():
    runs_standard_benchmarks = {
        'cifar10': {
            'Joint': ['3b6f760d008f4384a27665672675e725', '71d1092f9e68468bb123d0256975309e', '85a5b63e886945e4b6774e2a830eaef1', '8381a5e1de484387b848a39b651dcf79', 'd45dac0143cf4ba58bed0c5dca36064d'],
            'Finetuning': ['4b6e3d6ca75a4129ab728bbf6a454890', 'ee5667d673b140e9b961424e60b3f558', '863c9dcb11794f0bba1db5c4a9ac4e85', 'e0c8dab9da0d45338f434b8f9059745a', 'bccee33a55ff4f0fbc69aec8f0979d1e'],
            'oEWC': ['9170e110c5aa45a887d9e173ca992c5c', '0b12da4a679740a68dd80bcb9dad762e', '213e3865ba334d879dde7dbb3a88a647', '903e932937ba4c6693d160501141fcd8', '0d921da25a0549c78db5864f561bf3cd'],
            'SI': ['1e9dddde0acb41bea68c4e6492274f0c', '3d7a1255bb134f07922c7e03bc451bbe', 'fdda8ff328d0488a94c2a9745cc64de6', '9c2bf0eb857049df9d708da46a80e539', 'e9c0a51ab2e54fd495fda1c60100b269'],
            'iCARL': ['9d536b3efd7d4236b6809c305f722372', 'bb00333c8bca4edab7bb588b7a9f3cb6', '00113d857ce545f58090f69d3698cf9a', '6b09deb643a545769974fcc707a64fe5', '2584779138b74ac5a213e78507705696'],
            'GDumb': ['4103e7810e134b1fade5dd3203e309f0', '5d24534e2b6f464196ecb7bd7737e02d', '0bbbb5f9ff03474ebfb3b332c100b579', '6873da88a1cf400a861f95b10acd0a5e', 'f5bcad61594542ab825f87f1cd61cc92'],
            'ER': ['cf0a90027e2349af93c7609109af230a', 'e46a089460084f4aa4be3d2429e6f139', '26f1325aca9543f4a755b806503d3287', 'ed78073b3b8e4f019a3909e37fffa765', '542bbd7ebee74580ae450f2f30c41986'],
            'CLeWI+ER': ['117a9cf140a14faeaf411ea2da962651', 'ba919ae661e44838b899775a62ec5e42', '52e5f8c4a0424b1b85c2d04dfe20bebd', '632c1106489444878154119766e91f96', '67bfbd0366be4c5b9b9538b2e5a63118'],
            'aGEM': ['f89dc689961f486f99f3525d8f1026d6', '17c7754580164cd3a7d31c4a006f6be8', '1ec7253b84de474bacc4171e1740aecf', '97f52bafaa3c4c92b19a62ccb8e5d86e', '44a69af9a083402da24b69845b479ded'],
            'CLeWI+aGEM': ['88ced88ecf5f48a383cb013d38fabfb9', 'c7e566017e8f42e88ad441ac1a829ff6', '0af38d5ab4ee4bac8f7fb36624e83c3a', '57109cb02aa14e69a0a821835f70a54a', 'b2710fd1dd804655a60bf6046d8df33f'],
            'ER-ACE': ['d5d50499b90047e58e3c8420273c6945', '72290111cb624b868362323a2ba34116', 'ba73b2cbaa8a44a09fa1abe928ac16bb', '82fc09032af44d12942bf8be1c63dc3e', '0a1c80f40d234080ac4c9f33354d8490'],
            'CLeWI+ER-ACE': ['9c98c31901254ef78a30a5f788907bd5', 'cfb7a439370b4977ac7b0b2821c0a88f', '5e6a47a34cf84726b6f9cc251279f874', '9a589bc0f805458584b14c61cd2e35e4', '828766518af943269723f8e3778f8ecd'],
            'MIR': ['1a564fd244a444eea9594f3b589b2f8f', 'dbef193a7a9944878f71f4bcd71b47b8', '1894b250f4ea42f5a4a6a980c713c6a3', 'cad047f86ed94d10907732367de201ff', '991c98a93e2c4f4696704aaafcea3c3d'],
            'CLeWI+MIR': ['2bc3a26a008440edbff21507fdd8e78d', '1248cfca39cb42939d84b85bda9b0a81', '8d7d4625002943649d835dac5f9deca7', 'f6edc1f46f57457c942df258536c433d', '2edd8b34f37143d6b296458dc8de2b6c'],
            'BIC': ['5ec300bbd2e9479a93359b0de9d4b2b5', '586e8b80400445419ed166e90ba8d48d', '66b5dd76828e47f4baec353d5a584552', '133703a6558444cbb2bd54de271e067b', 'ab754288b1e84a93b7934d0d9d2a3fe5'],
            'CLeWI+BIC': ['f07c9ddd23e54862805941c036441526', 'd4d6f98a347f4ca29db50f8fbd660f0d', '814412b1311c41709dea9f642810b605', '90b2715d86d342cb8d99e787d6479ed4', 'a2091c7671c3461c9b277f86fc3e42c6'],
            'DER++': ['f0332d41ac47409a98824ca78a08f157', '050c7026724a40d6af1c7c309f48afb0', '1060653fa198413abbaf470b722b1668', '95ebe92010b349fbafd4c0f3e079015a', 'eb3601785bb74c108dc15ef9fbda0d2e'],
            'CLeWI+DER++': ['f4f01c401e534be4a5fdd5865c72a76b', '002842ef491f4a2cbcf1d3758eea82a0', '907467efd7f94a1cbb72cdd31ef62b2b', '35c0745f43dd495a908d6ffe5f34bea8', 'f63631969e7042e4a62805d9872d3343'],
        },
        'cifar100': {
            'Joint': ['340276f083b64cc49dccd2d69c9ad66a', 'ef826a5b9e334f4b92cc3fa695847b8a', '663bcd4b333248a5b559a01e376585dd', 'b008c7833f404dfca446753cefdcdf1e', '4f53e6af47174940b04fa70d618af51a'],
            'Finetuning': ['17b3122b6a5f420b9661e004af2a0588', '93980b3382fd438fb84635aead2e89df', '7b026e572ccd47938ce600c8b76248a8', 'f7f8eba20dbb454d80df3c4f2fa00c8d', '5d0ab7d56d3746d08d5d7c98cf3232ff'],
            'oEWC': ['191280d66c724ba390016d49b4544ce8', 'befd7e8a18604801a486d69f45d8e0ac', 'b0c5e9759ad64040ac330ddf1be4382a', '0d2b0c75b79e4a0997cc1b0168300965', '7e39d2481c5f4c3ea57e97df906006aa'],
            'SI': ['47d5e185871d4f8184f80ae580807906', '75d48d891fe64254b4bc5e252c5aa7c1', 'db800f218e5848b8b7cb17978bd85f84', 'fc99b0f9ff0f41329966ff78b507a148', '0f10cd9068fe4654ac90a25fd3a2d97a'],
            'iCARL': ['4dd5033dfda04f6a98b646774b1d8229', 'b3639bf984824afd88e9aec812290dce', '41737907295f43cfa6696761f2455fad', 'e9e2562273844d2f9ac8a8c8ccb507f4', '84de7552699e4beda7196121027622c5'],
            'GDumb': ['cdd60faa69da45549ed0eb9804c0bbfe', '887164556a234e228bb38017abe44ab6', 'b92f47cfa43d4363a37330ddaf23033c', '81a91133ccda4800ae86b01315b7a0ad', '692af0b23b5d416380421876fc4b591c'],
            'ER': ['46fda71735f64c8da98d050960dd3cd9', '7835e0307ca143689620fced42f62c9d', '2239aa1376dd4467852af4347adf4811', 'fe4136cbe534423ab257fcf0e8914300', 'be01e5f0683f410885922ef516eb5591'],
            'CLeWI+ER': ['cf7c6b8b65b04450a468d2f2fa0e9ac9', '5b0b0cec489644b789105d0852223c02', '189b17529d5d49dfb549d1c3d3256306', '66c13b0edaff45eca4c5d79facce9c08', '60ccb27498024749a1292d1ab2ec66e1'],
            'aGEM': ['66eb4c88e25b42de81016a2bbc2ada2d', '36e7ae355bd440d2b7de47d850ebc464', '7315f6bad23d4ff086f2a4340cc335db', 'f4a9b57dc701435c8fc1595547cd6ffd', 'db8fe29d318a49b08d886ac21d83c6dd'],
            'CLeWI+aGEM': ['a3cc47c1ca9c41ebbcda996c59eecfdd', 'ad3953382b694924981c5d8a6d906bed', '96ab9dd7d484433aaacf6a8ef07939fa', '5d292d3e82e44cafab3a79281b006b9b', '4bf6ca4fad20479283befa1acdc0ac63'],
            'ER-ACE': ['734e25c074514804a0c012e05d9b5450', '1b6cd95859c94471b3cdbed16d242306', '77d24d4c2eeb4b3d8c8566ec5494ac98', '26cd9495cfda4a4984b2532f8d6a80e7', 'eeb81e6993164c8090da796abc8f9ee1'],
            'CLeWI+ER-ACE': ['30f722588dbe49179757ee7f34b0d1f2', 'ef9737eb24cf421f8116616cb399bef9', '1e5cf3ea5d384dbbaf7ae67840aaca3b', '2f513b260fd84454abaca0c32ee554f9', '2e53e881984d46a8842f71708fb811bd'],
            'MIR': ['3e2d37d427e04f03855243fefdc6e1fd', '70cbe7616f7a433c891931594d5fd1af', 'e23a3a90090149ea91c1cecd4bcdc404', '20b096e7d73b46b4a5867477eef29a63', '2413e62de8bb45c3aabe0da54189ab52'],
            'CLeWI+MIR': ['28a3fdd20fa24a1cb786450d8fcb83e8', '3f8698c62abd4a7c8f2b4ae6557d7f50', '91d6c988ad444a5aa5fa54ab83778f46', 'ab8047d14e1240d198b9c4604852501d', 'fac71bfdcfd544db9ee086a281ca769f'],
            'BIC': ['1cd4ab0fff654d66952518480bace581', '4396afc6f0ad4da6b95c6e7508f68cc3', '725ad3cc7299453aac8798ffc35676dc', '5d5266c29a80430facfe17080bfb2edb', 'ba359e2d933e471087fea26956cafd6d'],
            'CLeWI+BIC': ['2c59ec45a5734d7988a9bd5338d0ba12', '59eecbb62f2342f79fdcce3448fbc0d8', '4a8e58cbd2de4a6b9c945ace09d59315', '98bddddfe0f24b48989bde75d82ad1ac', 'e44efc8096e040ae97fd8cf739f9be62'],
            'DER++': ['ae8b5183dcf2473098e4d6a2c040f8d7', 'a8002e911d8343f99aec7a59d67c34bb', '186e1034cd914a658c83d2c1c076df27', 'ca4ab2fde2a54595b308db566fc2753a', '142ce6e4e98d4c94b050c1f1acc55601'],
            'CLeWI+DER++': ['ec55afa0eba747f39390748811e09ffd', 'a55184ff8497433ca3df7f78d7bed368', 'a539e8526d994e60a16aeba62a13fd0f', '352f030fbf9f4b9d83e8782c63162ddc', '5c63ddbacb714799a15bbb4ec0265b3e'],
        },
        'tiny-imagenet': {
            'Joint': None,
            'Finetuning': ['bb467533c51d4f4d8b6f12c86c6b4a3e', 'e7b37e974309450b895df0d9f4b6e4b9', 'cf68367c6c3c401eb3b6d99b07fd3b75', 'b112ed4e336f465bb6072ad84b837862', '5fc622e1e37d4217b3b2a3ecb66e23b8'],
            'oEWC': ['127dc1061a01424ebdcc849ea0d647b5', 'b5a6d3367aa9471bb06e5c77b96a58bc', '7445317343ec4ae68a087a25646d6eed', '4a0a19b0f3674992a21cf7eb9662ef27', '8e92f75b5d104899b66ce5c22f2a9078'],
            'SI': ['1aed8e7dcf994c5cb26ae2316ab42ac6', '51d05c34b7c94b1e9d434a40155ceaef', 'd3342607c59345978425f2b0dacf4f61', 'f827a662b7ac413a806cb224d04f4f99', 'b275260ed39d4d5ab0562e337bce16e2'],
            'iCARL': ['a2cd5ca6d9714c388679aac0ee035854', 'efb84e6889f34ee2ba05195bfefb0270', '78bf2fd410804526b73376edef751a59', 'c2b5b86a92b942fa953b4f3836aee9a4', '3c995cbf8ce2428090140cbc6cb480dc'],
            'GDumb': ['d1c4078a7b874e0091b682f1719aa70d', '9dd9644b1411452db420925c90b2dd25', '842ae1feffe04599b32d7155955e6be5', 'e42443f5d9664564835fa8eb1e9dc357', 'bb3c73b2469647bf8d91ca007344a293'],
            'ER': ['846876270e014703afbcfccf4d983037', '71e7f27eef0b4ec6a0ec211c4a32591d', '0f21e2b127d14c5884a68b970f1a3d44', 'a1adc252c6fb4daa8283334d4581127a', '291315fb5a684937b9835b1c5509b299'],
            'CLeWI+ER': ['23b7b115930f49319437bb924981120e', '54da4d2a2e50404b86db2aa360719567', '09e1a64120bb4acb8ef59774082b80d5', '170335b48a0f44b4a634de5848d841f0', '8bab656357e2467597816490d9182168'],
            'aGEM': ['573f2ad340e646afad6823a5d6be0c73', '563d5d2044ef446bae3a6e0986193411', '1dbd9df4d19b4b429aff401b0a9e1c55', 'e08abf607aac41c2967fd0b295410648', '6238b6d4c4d344789c61edc8425478f4'],
            'CLeWI+aGEM': ['29bcaa26526a4f9d90a1908f37eff054', '7ad40b999b704e838243c87843a58e80', '8a217b9273ce4d44896103df7f9edffc', '1072739b4b4b40178568840044cee9ed', 'd6af7766331d486d8c3465fba1e18dae'],
            'ER-ACE': ['a2fb4aed0153415588bb2f02f86087f6', '4743e2fc63bf402993de8bc781738f92', 'c2e2ee0781ac41a6a8b3f2ea4faa965f', '7dddc4e05c3748609e79a277c5aec3c0', '576af1e4d2ed4dac922ca71ef6de6c9e'],
            'CLeWI+ER-ACE': ['62ff662c013f4ddf9315c506748ef11a', '120387133a6c4d448b120ee8318641a4', '41f911881f47406aab483a9aa4635bc3', '0d654576792043a6aa0aaf23b0ebdb52', 'd504d98c3199434ba339bc0a8fb5d7c7'],
            'MIR': None,
            'CLeWI+MIR': None,
            'BIC': ['f77def6d9cae422eaa46fa802bd0ee4d', '3ad493457e8842b793eec3e0fed0fd8b', '2ab72dd897044f1eba9a2030670e746d', '92a5abee359b45d19ee2a70f973460b4', 'b4baeb8326084acf9585316ce7ce96cc'],
            'CLeWI+BIC': ['d6f22dc79ab34f3795600de91f7bdf35', 'a7db5b0a8c2348f9aeb093f88d5644b6', 'f0ae75ea46624db1aef0e99f4cf716d7', 'fd225489ca674152a886db17368b3ff2', 'c89efaa4f8f042d59c80a3c4d5c98a0f'],
            'DER++': None,
            'CLeWI+DER++': None,
        }
    }

    assert runs_standard_benchmarks['cifar10'].keys() == runs_standard_benchmarks['cifar100'].keys() == runs_standard_benchmarks['tiny-imagenet'].keys()
    algorithms = list(runs_standard_benchmarks['cifar10'].keys())

    mlruns_path = '///home/jkozal/Documents/PWr/interpolation/weight-interpolation-cl/mlruns/'
    # mlruns_path = '///home/jedrzejkozal/Documents/adversarial-computer-security/mlruns/'
    client = mlflow.tracking.MlflowClient(mlruns_path)

    dataset_experiments = {
        'cifar10': '444536103888854616',
        'cifar100': '675415310966171557',
        'tiny-imagenet': '757344672409704114'
    }

    table = list()
    for algorithm_name in algorithms:
        row = list()
        row.append(algorithm_name)
        for dataset, n_tasks in zip(('cifar10', 'cifar100', 'tiny-imagenet'), (5, 10, 20)):
            run_ids = runs_standard_benchmarks[dataset][algorithm_name]
            experiment_id = dataset_experiments[dataset]
            metrics = calc_average_metrics(run_ids, client, experiment_id, n_tasks=n_tasks)
            row.extend(metrics[:-1])

        if row[0].startswith('CLeWI+'):
            previos_row = table[-1]

            # print(row)
            for i in range(1, 6, 2):
                # print(row[i])
                # print(row[i+1])
                # print()

                if row[i] == '-' or previos_row[i] == '-':
                    continue

                difference_acc = float(row[i].split('±')[0]) - float(previos_row[i].split('±')[0])
                row[i] += get_difference_str(difference_acc)
                difference_fm = float(row[i+1].split('±')[0]) - float(previos_row[i+1].split('±')[0])
                row[i+1] += get_difference_str(difference_fm, bigger_better=False)

            # print(row)
            # exit()
        table.append(row)

    tab_latex = tabulate.tabulate(table, tablefmt="latex", headers=['method', 'acc', 'FM', 'acc', 'FM', 'acc', 'FM',])
    tab_latex = tab_latex.replace('\\textbackslash{}', '\\')
    tab_latex = tab_latex.replace('\\{', '{')
    tab_latex = tab_latex.replace('\\}', '}')
    print(tab_latex)
    print("\n\n")


def get_difference_str(difference, bigger_better=True):
    if bigger_better:
        greather = 'Green'
        lower = 'Red'
    else:
        greather = 'Red'
        lower = 'Green'
    difference_str = '{:.2f}'.format(difference)
    if difference >= 0:
        difference_str = '\\textcolor{' + greather + '}{' + f'+{difference_str}' + '}'
    else:
        difference_str = '\\textcolor{' + lower + '}{' + f'{difference_str}' + '}'
    difference_str = f'({difference_str})'
    return difference_str


def varing_n_tasks():
    runs_n_tasks = {
        '5 tasks': {
            'Joint': ['0b3a5c552bf840b7bf3b283d7adda616', 'a5b21afc059f48cab2595679d2b26d0a', '763b69704b73425e97e4771586c3ee78', 'f02f0b88fe134600820524374ab5197e', 'e2da958ae0274d5a94591c6db34ab998'],
            'Finetuning': ['06512a6ae08d44f4887a0286a7d92211', '5bd6ede41509449e9f877330388d8083', 'e14425b91aab44518d0c774813bdf962', 'afbc8e2906bf4feebfa32d024c92b8e4', '19e9676976f04e3f9f2b5521c9e19ada'],
            'oEWC': None,
            'SI': None,
            'iCARL': ['7dc675684dfa4f8cacf8aeb1cc09ffc1', 'c9ed7dd02f5649b3b9792c40576832c7', 'f0bda601517343aaaba222367d242233', 'c604e22e957f4ccf98d251babcabc60a', '77dd9b747c4d45aab6d8011540eb4fad'],
            'ER': ['49e3cac631bf487599e3ec0387bf0560', '5ebebc19adf14b5698e8897fe7d8fbde', '77150474837640baa21fd93d8a9ebe80', 'ce32bab6ad9a4a47b03d25e2dbaf9f32', '4d1f067b3c12432ead29617eeaa6d173'],
            'aGEM': None,
            'BIC': ['d422de3abd7d411d8e79e6438203c869', '60907987ce48472791791bd13b96fe1c', '2cd2dfe8d6c84ee8878730fae7a067de', 'a0367c72efe94198a24700acfe2ee89f', '7207fc72b94e48878338d4aaec71dd1c'],
            'ER-ACE': ['8b154fad156b4c0cbbb1b8bc58bcdc3f', '30333191db024696bfa248cc57e92381', '07447ad47d9748ff892ccad06d3c38e7', '02984412768841aeb5b9d51d0c280018', 'eddb1bdfca164937a2ceeef1272ceabd'],
            'GDumb': ['8b5ec60afcc14f0489d3861c66082c51', 'e53a5538208946b9b2c21b492eeb2af4', 'b53f529c1d2d4f26927c84445b7309ab', '6205d45f81e040e98680be7af7f07e88', '10c69bba417a49c38b07c8bc6d81fb49'],
            'MIR': ['1137061e495c406fbf532ce2c0f2a9f9', 'f24325e75f314d8989ca850a9003e84a', 'a46fcde138574818920d31f9a47ab45e', 'e8ced848a66343f9b00ee16d3eea8167', '5b008bcaf06544099ebe98d67d43313f'],
            'DER': ['fd72fc1b81f84b8caeaf7acb207a2f97', 'fa27e138808e48e68f00d23c02b03507', '61dc397fe81f488e98725ee37aacaabe', '7aabdcca65a44d73a5015c15f8cabd25', '217fb45437f4425a94c70017aa26901c'],
            'xDER': ['fe27a2cd3a414b498d984ee9d6019952', '5dc8077a643a40e19b21e6337991b488', 'b5ec01658cf14b81b824ba916782c9a8', '0d7c109accc645e08f1fcd0594f4b6f1', '0f30523a31ad4bedbb831920cb9cd530'],
            'CLeWI+ER': ['0fa6f70ffe104b69ac8899a8ccbacb7f', 'f59a01c79b7a477e8b3866be57ee431b', 'cb276ba0a0f04fe6bfc0be5c5a526110', '4088b64d49814c848ca0f2a2d3e20f51', '75126b5c9c4645beae9204ab7d1ba667'],
            'CLeWI+aGEM': None,
            'CLeWI+ER-ACE': None,
            'CLeWI+MIR': None,
            'CLeWI+DER++': None,
        },
        '10 tasks': {
            'Joint': ['340276f083b64cc49dccd2d69c9ad66a', 'ef826a5b9e334f4b92cc3fa695847b8a', '663bcd4b333248a5b559a01e376585dd', 'b008c7833f404dfca446753cefdcdf1e', '4f53e6af47174940b04fa70d618af51a'],
            'Finetuning': ['17b3122b6a5f420b9661e004af2a0588', '93980b3382fd438fb84635aead2e89df', '7b026e572ccd47938ce600c8b76248a8', 'f7f8eba20dbb454d80df3c4f2fa00c8d', '5d0ab7d56d3746d08d5d7c98cf3232ff'],
            'oEWC': ['191280d66c724ba390016d49b4544ce8', 'befd7e8a18604801a486d69f45d8e0ac', 'b0c5e9759ad64040ac330ddf1be4382a', '0d2b0c75b79e4a0997cc1b0168300965', '7e39d2481c5f4c3ea57e97df906006aa'],
            'SI': None,
            'iCARL': ['4dd5033dfda04f6a98b646774b1d8229', 'b3639bf984824afd88e9aec812290dce', '41737907295f43cfa6696761f2455fad', 'e9e2562273844d2f9ac8a8c8ccb507f4', '84de7552699e4beda7196121027622c5'],
            'ER': ['46fda71735f64c8da98d050960dd3cd9', '7835e0307ca143689620fced42f62c9d', '2239aa1376dd4467852af4347adf4811', 'fe4136cbe534423ab257fcf0e8914300', 'be01e5f0683f410885922ef516eb5591'],
            'aGEM': ['66eb4c88e25b42de81016a2bbc2ada2d', '36e7ae355bd440d2b7de47d850ebc464', '7315f6bad23d4ff086f2a4340cc335db', 'f4a9b57dc701435c8fc1595547cd6ffd', 'db8fe29d318a49b08d886ac21d83c6dd'],
            'BIC': ['1cd4ab0fff654d66952518480bace581', '4396afc6f0ad4da6b95c6e7508f68cc3', '725ad3cc7299453aac8798ffc35676dc', '5d5266c29a80430facfe17080bfb2edb', 'ba359e2d933e471087fea26956cafd6d'],
            'ER-ACE': ['734e25c074514804a0c012e05d9b5450', '1b6cd95859c94471b3cdbed16d242306', '77d24d4c2eeb4b3d8c8566ec5494ac98', '26cd9495cfda4a4984b2532f8d6a80e7', 'eeb81e6993164c8090da796abc8f9ee1'],
            'GDumb': ['cdd60faa69da45549ed0eb9804c0bbfe', '887164556a234e228bb38017abe44ab6', 'b92f47cfa43d4363a37330ddaf23033c', '81a91133ccda4800ae86b01315b7a0ad', '692af0b23b5d416380421876fc4b591c'],
            'MIR': ['3e2d37d427e04f03855243fefdc6e1fd', '70cbe7616f7a433c891931594d5fd1af', 'e23a3a90090149ea91c1cecd4bcdc404', '20b096e7d73b46b4a5867477eef29a63', '2413e62de8bb45c3aabe0da54189ab52'],
            'DER': ['ae8b5183dcf2473098e4d6a2c040f8d7', 'a8002e911d8343f99aec7a59d67c34bb', '186e1034cd914a658c83d2c1c076df27', 'ca4ab2fde2a54595b308db566fc2753a', '142ce6e4e98d4c94b050c1f1acc55601'],
            'xDER': ['f248a2f2900b4f0f9682e4629d49930b', 'e7f20f8c086243e29699307e782dd116', '66c73464bf1048cf9d0d454814f6607e', 'b9b03222b2b8470c9cc5303510dda447', '1e2937c86b54484cbf88569da7b00ba9'],
            'CLeWI+ER': ['cf7c6b8b65b04450a468d2f2fa0e9ac9', '5b0b0cec489644b789105d0852223c02', '189b17529d5d49dfb549d1c3d3256306', '66c13b0edaff45eca4c5d79facce9c08', '60ccb27498024749a1292d1ab2ec66e1'],
            'CLeWI+aGEM': ['a3cc47c1ca9c41ebbcda996c59eecfdd', 'ad3953382b694924981c5d8a6d906bed', '96ab9dd7d484433aaacf6a8ef07939fa', '5d292d3e82e44cafab3a79281b006b9b', '4bf6ca4fad20479283befa1acdc0ac63'],
            'CLeWI+ER-ACE': ['30f722588dbe49179757ee7f34b0d1f2', 'ef9737eb24cf421f8116616cb399bef9', '1e5cf3ea5d384dbbaf7ae67840aaca3b', '2f513b260fd84454abaca0c32ee554f9', '2e53e881984d46a8842f71708fb811bd'],
            'CLeWI+MIR': ['28a3fdd20fa24a1cb786450d8fcb83e8', '3f8698c62abd4a7c8f2b4ae6557d7f50', '91d6c988ad444a5aa5fa54ab83778f46', 'ab8047d14e1240d198b9c4604852501d', 'fac71bfdcfd544db9ee086a281ca769f'],
            'CLeWI+DER++': ['ec55afa0eba747f39390748811e09ffd', 'a55184ff8497433ca3df7f78d7bed368', 'a539e8526d994e60a16aeba62a13fd0f', '352f030fbf9f4b9d83e8782c63162ddc', '5c63ddbacb714799a15bbb4ec0265b3e'],
        },
        '20 tasks': {
            'Joint': ['aeea61c15292433e9b981c00873ee87f', 'b42b8405295c44928ba111d904265144', 'e59e9f24c8cc459d85b6bf0befd4caa8', '73259b1384904285bbc87ad6c372572a', '80c2683062ce4481ada44bcf99ac0dd1'],
            'Finetuning': ['bb644c8bdd8c47d89d14e1692e3c2c8e', '125a5d28aadd4e66839e173d8942a515', '7c23e4a7c2dc4483833b67c7054c628d', 'e39f89770ece4badbfa1831745eb3200', 'da049146b71148688dcaf58ec517b4be'],
            'oEWC': ['8753f219b2344fe695a86e21b14078e8', '64b8d37a83cd48a98968d1ad3e519981', '5002e50e739647539be89f7995e2eb89', '7c91df1cd7614efc9831860b399468de', '3a40f7d5baaf46528727f5e55236bb02'],
            'SI': None,
            'iCARL': ['94de55f367924c248a45c5987df33524', 'c8e0f027b51c49648f69f0eb38509462', '610e0073957d49548b68791dd9246919', '7a08a1afe0c741f6a5e893681aa9f8a2', 'de486b95a6c74bfda0a9dd1dd1ec246d'],
            'ER': ['b493ab8e54004750896dbf9cebf0b609', '1951a520b7974b7c9a6d2e26f70f1d3c', '997ecc0f525f4071a74f0b216e9f39c8', 'a8513f0c1db74676af3b9454f49e7a2f', 'e011a0cbc9854328b248df835ae5d178'],
            'aGEM': ['9405b16c0fe243c68c2c7d35e84aec17', '82b1def4d55c4719aa614728341a2154', '6da962ac0c1c46dbbfd1a58f650ac524', '5ff50f224d8b4f6190f81aaca22a9e05', '1855d3054a234e8f87b30b72b1ca7495'],
            'BIC': ['3baf04da53404ca4bbcc0506c90545bb', '5b0b5502cc82427f80df476408ce7da5', '12d168011906473aa728600e672f91d8', '1e686f8e6e4e4c96b113adfe1547e01c', '25c878dddec74378b069173f3caf503f'],
            'ER-ACE': ['a7e438cff0b64590a1b51e421bc200fc', '1b04f1d16d584333826c3c29d277633b', '459fb22963c04042800b2492307c208d', '9f2060e2ea934c109d5447895fddbfcc', 'a41d246aac464288bbcd990e602c276b'],
            'GDumb': ['ff544949cdda41dcb17666474f40db51', '330650c362984bd2abf34d6fbad93366', 'c3e0e9ddc19f4e1c8368411a54b31773', 'd11cb963b54c4a64ba58c50f86651eb6', 'f1389212c51b455eae5d57f8eeac01d0'],
            'MIR': ['c73632f271924e2eaaea438cc59745bc', 'f349082c209548ac9dda7ab7fa9e0738', '404cff1ecc344b3cb9290fe1da26408e', 'a36a28e3298a49dcadf83bf946975de3', '205f73a788134b35a80590338e98a990'],
            'DER': ['cd06015c7db24f308176ed5a2bb632d9', '6946c90170914e319cd522c2563995a8', '22fb4bca74c0441d8fc7314da250b4ba', 'df57945b9e304e94b6eb76c61e9778ac', '3157516230564a7887a4a4164843e12d'],
            'xDER': ['f4510037fdbf409d9a39c633d9961f33', '6279780737c64700af2dd780637300f4', '29f2ff8dba584728ac379c76e54341ab', '95383a9572b94a0180fa62dfb4bb01b6', '27b6ab60af294ce3a40ac56a33ffbe1b'],
            'CLeWI+ER': ['7157360c5e63454d9dd11a794f7a4319', '98284fb1575e4863a79b1e3a2b9d2ab6', '4cfed79122ab440aa81d5f4f1006b70a', 'bace4c4d02c24be2b923e5d4d8e84289'],
            'CLeWI+aGEM': None,
            'CLeWI+ER-ACE': None,
            'CLeWI+MIR': None,
            'CLeWI+DER++': None,
        }
    }

    assert runs_n_tasks['5 tasks'].keys() == runs_n_tasks['10 tasks'].keys() == runs_n_tasks['20 tasks'].keys()
    algorithms = list(runs_n_tasks['5 tasks'].keys())

    mlruns_path = '///home/jkozal/Documents/PWr/interpolation/weight-interpolation-cl/mlruns/'
    # mlruns_path = '///home/jedrzejkozal/Documents/adversarial-computer-security/mlruns/'
    client = mlflow.tracking.MlflowClient(mlruns_path)

    table = list()
    for algorithm_name in algorithms:
        row = list()
        row.append(algorithm_name)
        for n_tasks in ('5 tasks', '10 tasks', '20 tasks'):
            run_ids = runs_n_tasks[n_tasks][algorithm_name]
            n_tasks = int(n_tasks.split()[0])
            acc, _, _ = calc_average_metrics(run_ids, client, '675415310966171557', n_tasks)
            row.append(acc)
        table.append(row)

    tab_latex = tabulate.tabulate(table, tablefmt="latex", headers=['method', '5 tasks', '10 tasks', '20 tasks',])
    tab_latex = tab_latex.replace('\\textbackslash{}', '\\')
    tab_latex = tab_latex.replace('\\{', '{')
    tab_latex = tab_latex.replace('\\}', '}')
    print(tab_latex)
    print("\n\n")


def interpolation_coef():
    runs_dict = {
        '$\\alpha$=0.1': ['6c18801f786641a29744253b16384999', '92a29446a8b74091894bdc5c3a42922a', 'b70c33c69cdb487c95be3472869b6009'],
        '$\\alpha$=0.2': ['b1e9e5fc86204e63b7d01e93a5cb1b3c', '36ecede7b11d40848a26ae8365e21730', '76816403a395414ea009d1f26f595398'],
        '$\\alpha$=0.3': ['b1c44f969b2046938906b826556eec9b', '436da7d90acd4fe5ba3712100cb9081e', '7f46a1e00eb7499dbc987ed8a44e56ad'],
        '$\\alpha$=0.4': ['85a739a6bf914d0b8afaab8f1fae6ccd', 'f9bcb6f16b71457d8c8b33555b267a65', 'b3618ad747a545fb910bea58614a09f7'],
        '$\\alpha$=0.5': ['cf7c6b8b65b04450a468d2f2fa0e9ac9', '5b0b0cec489644b789105d0852223c02', '189b17529d5d49dfb549d1c3d3256306'],
    }

    algorithms = list(runs_dict.keys())

    mlruns_path = '///home/jkozal/Documents/PWr/interpolation/weight-interpolation-cl/mlruns/'
    # mlruns_path = '///home/jedrzejkozal/Documents/adversarial-computer-security/mlruns/'
    client = mlflow.tracking.MlflowClient(mlruns_path)

    table = list()
    for algorithm_name in algorithms:
        row = list()
        row.append(algorithm_name)
        run_ids = runs_dict[algorithm_name]
        if algorithm_name == '$\\alpha$=0.5':
            experiment_id = '675415310966171557'
        else:
            experiment_id = '654603390611542524'
        acc, fm, last_acc = calc_average_metrics(run_ids, client, experiment_id, n_tasks=10, digits=2)
        row.append(acc)
        row.append(last_acc)
        row.append(fm)
        table.append(row)

    tab_latex = tabulate.tabulate(table, tablefmt="latex", headers=['alpha', 'Acc', 'Acc_T', 'FM',])
    tab_latex = tab_latex.replace('\\textbackslash{}', '\\')
    tab_latex = tab_latex.replace('\\{', '{')
    tab_latex = tab_latex.replace('\\}', '}')
    tab_latex = tab_latex.replace('\\$', '$')
    print(tab_latex)
    print("\n\n")


def calc_average_metrics(dataset_run_ids, client, experiment_id, n_tasks=20, digits=3):
    if dataset_run_ids == None:
        return '-', '-', '-'

    acc_all = []
    fm_all = []
    last_task_acc_all = []
    for run_id in dataset_run_ids:
        acc = get_metrics(run_id, client)
        acc_all.append(acc)
        fm = calc_forgetting_measure(run_id, client, experiment_id=experiment_id, num_tasks=n_tasks)  # TODO fix logging num_tasks in experiments
        fm_all.append(fm)
        last_task_acc = get_last_task_acc(run_id, client, experiment_id=experiment_id, num_tasks=n_tasks)
        last_task_acc_all.append(last_task_acc)

    avrg_acc, acc_std = rounded_reduction(acc_all, digits=digits)
    acc = f'{avrg_acc}±{acc_std}'
    avrg_fm, fm_std = rounded_reduction(fm_all, digits=digits)
    forgetting = f'{avrg_fm}±{fm_std}'
    avrg_last_acc, last_acc_std = rounded_reduction(last_task_acc_all, digits=digits)
    last_acc = f'{avrg_last_acc}±{last_acc_std}'
    return acc, forgetting, last_acc


def get_metrics(run_id, client):
    run = client.get_run(run_id)
    run_metrics = run.data.metrics
    acc = run_metrics['mean_acc_class_il']
    return acc


def rounded_reduction(metrics, digits=4):
    metrics = np.array(metrics)
    avrg = metrics.mean()
    avrg = round(avrg, digits)
    std = metrics.std()
    std = round(std, digits)
    return avrg, std


def calc_forgetting_measure(run_id, client, experiment_id, num_tasks=None):
    run_path = pathlib.Path(f'mlruns/{experiment_id}/{run_id}/metrics/')
    if num_tasks is None:
        run = client.get_run(run_id)
        num_tasks = run.data.params['n_experiences']
        num_tasks = int(num_tasks)

    fm = 0.0

    for task_id in range(num_tasks):
        filepath = run_path / f'acc_class_il_task_{task_id}'
        task_accs = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                acc_str = line.split()[-2]
                acc = float(acc_str)
                task_accs.append(acc)

        fm += abs(task_accs[-1] - max(task_accs))

    fm = fm / num_tasks
    return fm


def get_last_task_acc(run_id, client, experiment_id, num_tasks=None):
    run_path = pathlib.Path(f'mlruns/{experiment_id}/{run_id}/metrics/')
    if num_tasks is None:
        run = client.get_run(run_id)
        num_tasks = run.data.params['n_experiences']
        num_tasks = int(num_tasks)

    filepath = run_path / f'acc_class_il_task_{num_tasks-1}'
    with open(filepath, 'r') as f:
        for line in f.readlines():
            acc_str = line.split()[-2]
            last_acc = float(acc_str)

    return last_acc


if __name__ == '__main__':
    main()
