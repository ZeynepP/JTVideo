import os
import os.path as osp
import json
import shutil

def create_enumarate_labels(fold_folder):
    labels = set()
    with open(fold_folder, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            file_name = str(obj["id"]) + ".mp4"
            label = str(obj["topics"][0]["label"]).replace(" ", "_").lower()
            labels.add(label)
    return list(labels)


def prepare_data_by_fold(data_path, video_path, fold_id):
     #target_classes = create_enumarate_labels(osp.join(data_path, fold_id, "train.json"))
     #classes_count = {k: v for v, k in enumerate(target_classes)}
     classes_count = {'santé': 0, 'culture-loisirs': 1, 'société': 2, 'sciences_et_techniques': 3, 'economie': 4, 'environnement': 5,
      'politique_france': 6, 'sport': 7, 'histoire-hommages': 8, 'justice': 9, 'faits_divers': 10, 'education': 11,
      'catastrophes': 12, 'international': 13}
     print(classes_count)
     for file_type in ["test","train","val"]:
         fold_folder = osp.join(data_path, fold_id, file_type + ".json")
         output = open(osp.join(data_path, fold_id, file_type + ".csv"),"w+")
         with open(fold_folder, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                file_name = str(obj["id"])+".mp4"
                label = classes_count[str(obj["topics"][0]["label"]).replace(" ","_").lower()]
                print(file_name, label)
                if not osp.exists( osp.join(video_path,str(label))):
                    os.mkdir(osp.join(video_path,str(label)))
                try:
                    shutil.move(osp.join(video_path,file_name), osp.join(video_path,str(label), file_name))
                except:
                    print(file_name)
                output.write(str(osp.join("/usr/src/mp4",  str(label), file_name ))+ "\t" + str(label) + "\n")

         output.close()


prepare_data_by_fold("../", "/usr/src/mp4", "0")

# import fiftyone as fo
# dataset_dir = "./0"
# # Create the dataset
# dataset = fo.Dataset.from_dir(
#  dataset_dir, fo.types.VideoClassificationDirectoryTree, name='dataset'
# )
# # Launch the App and view the dataset
# session = fo.launch_app(dataset)

#{"id":201805414,"nativeId":"Se 38,18 brut Tf1.xls","docTime":"2018-09-21T20:01:47.07+02:00","media":"tv/tf1","title":"[Le trafic de crack à la porte de la Chapelle]","document":"adrien penser presque au pied du stade de france caché par le périphérique parisien chaque jour des centaines de personnes accède à ce petit morceau de terre pour acheter ce qu' on appelle la drogue du pauvre quelques draps pour des transactions discret certains ne reste que quelques minutes d' autres vivent ici jour et nuit quinze euros la galette de crack consommation en plein air nous rencontrons ahmed cinquante deux ans pour ne pas se faire arrêter par la police mais qu' est ce que vous faites alors à partir de ce moment là emballée dans du film plastique ahmed est accro depuis vingt huit ans maintenant à ce petit caillou blanc un dérivé de la cocaïne placé dans une pipe la drogue est inhalé en quelques secondes de ce menu tout d' une minute et vous sentez mieux dans ces cas là j' ai peur j' je peux pas la qui peut définir j' ai plein de gens qui me voient et je peux me nous vous voye crise de paranoïa hallucinations c' est très courant chez ce qu' on appelle les crackers isabelle a cinquante trois ans après plusieurs cures elle a replongé ses enfants habitent à quelques kilomètres d' ici elle ne les a pas revus depuis plusieurs mois ils nous ont donné une ici et je très rétive voile est en train de ils exigent la règle du jeu de la que je veux confirmer chaque jour isabelle prend pour environ soixante euros de crack comme beaucoup de femmes dans la colline tout en grande précarité elle dit devoir se prostituent minimum s' attend avant la galette des entretiens un homme alors vous devez reconnu qui rué à acheter les agret quelques heures plus tard quand nous revenons sur la colline présence policière pour encadrer un nettoyage c' est le terme des agents municipaux saisies tenter mobilier de fortune pas d' interpellation les dealers les consommateurs beaucoup ont juste traverser la route dix mètres plus loin le trafic a repris et ce trafic et de plus en plus visible de plus en plus nuisibles d' après le maire du dix-huitième arrondissement excédés il appelle l' à l' aide euh bon des effectifs et des effectifs nombreux pour assurer la sécurité publique pour faire en sorte que les habitants qui ne sont ni des acheteurs ni des vendeurs puis vivre tranquillement dans leur quartier et sans doute des enquêtes plus longue d' où vient le produit comment entrent sur le territoire peut -on l' arrêter dans les faits depuis quelques années le crack est de plus en plus visible dans la capitale sous les yeux d' enfants en sortie scolaire dans cette station du nord de paris un homme fumer sa pipe de crack scène fréquentes sur cette ligne même si officiellement la ratp parlent d' une situation qui s' améliorent un conducteur nous dit le contraire il y a encore dit -il signaler à ça non je crois qu' on arrive mais sont allés pour surmonter un avenir décent c' est mon bon mollissant avec quinze mille consommateur de crack en ile-de-france l' urgence c' est aussi la prise en charge dans cette association un millier de patients traités l' an dernier bien qu' aucun médicament ne permet de soigner à lui seul la dépendance au krach parfois c' est le social qui prime avoir un toit sur la tête euh avoir des ressources pouvoir renouer avec sa famille permet tout simplement d' abandonner euh on certaines consommations quoi le krach et c' est un autre problème se démocratise parmi les nouveaux usagers on compte désormais des cadres et les professions libérales une","notes":"Reportage consacré au trafic de crack dans un espace situé près de la porte de la Chapelle au bord du périphérique parisien. Les acheteurs et les vendeurs, plusieurs fois chassés par les services de police, reviennent toujours. Le journaliste interviewe plusieurs toxicomanes et un conducteur de métro anonymement. Il y aurait environ 15 000 consommateurs de crack en Ile de France.Commentaire sur des images factuelles, dont certaines en caméra cachée, et d'illustration, entrecoupé par les interviews de Eric LEJOINDRE, maire PS du 18ème arrondissement de Paris, de Elisabeth AVRIL, directrice de l'association Gaia Paris.","topics":[{"label":"Justice","type":"InaStats","score":1.0,"inferred":false}],"endTime":"2018-09-21T20:05:43.92+02:00"}
