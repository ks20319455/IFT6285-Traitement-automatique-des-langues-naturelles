from acl_anthology import Anthology
import csv
import random
anthology = Anthology.from_repo()

authors_dict={}
for collectionName in anthology.collections:
    collection=anthology.get(collectionName)
    for volume in collection.volumes():
        for paper in volume.papers():
            if(paper.abstract):
                for author in paper.authors:
                    author_full_name=(author.name.first or "")+" "+(author.name.last or "")
                    if(author_full_name not in authors_dict):
                        authors_dict[author_full_name]=1
                    else:
                        authors_dict[author_full_name]+=1

# Get the 10 highest values
num_classes=9
top_authors = sorted(authors_dict, key=authors_dict.get, reverse=True)[:num_classes]
cnt1=0
dict1 = {author: 0 for author in top_authors}
cnt2=0
dict2 = {author: 0 for author in top_authors}
with open('train-devinesqui.csv', 'w', newline='') as train_csvfile:
    with open('test-devinesqui.csv', 'w', newline='') as test_csvfile:
        train_spamwriter = csv.writer(train_csvfile, delimiter=';')
        test_spamwriter = csv.writer(test_csvfile, delimiter=';')
        train_spamwriter.writerow(['id','title','abstract','classe'])
        test_spamwriter.writerow(['id','title','abstract','classe'])
        for collectionName in anthology.collections:
            collection=anthology.get(collectionName)
            for volume in collection.volumes():
                for paper in volume.papers():
                    if(paper.abstract):
                        for author in paper.authors:
                            author_full_name=(author.name.first or "")+" "+(author.name.last or "")
                            if(author_full_name in top_authors):
                                num=random.random()
                                if(num>=0.2):
                                    train_spamwriter.writerow([paper.full_id, paper.title, paper.abstract, author_full_name])
                                    dict1[author_full_name]+=1
                                    cnt1+=1
                                else:
                                    test_spamwriter.writerow([paper.full_id, paper.title, paper.abstract, author_full_name])
                                    dict2[author_full_name]+=1
                                    cnt2+=1

# Open a file in write mode
with open('desc-devinesqui.txt', 'w') as file:
    # Write some text to the file
    file.write('Tache de classification ayant 9 classes visant a idenfitifer les authors des articles. On prend seulement les 9 auteurs avec le plus nombre d\'aritcle publiees dans toutes les conferences incluts dans le dataset .\n')
    file.write(f'Classes [{num_classes}]: {", ".join(top_authors)}\n')
    file.write('features: title, abstract\n')
    file.write(f'test-deviensqui.csv: {cnt2}\n')
    test_articles = ', '.join(f'{author} {dict2[author]}' for author in top_authors)
    file.write(f'Test: {test_articles}\n')
    file.write(f'train-deviensqui.csv: {cnt1}\n')
    train_articles = ', '.join(f'{author} {dict1[author]}' for author in top_authors)
    file.write(f'Train: {train_articles}\n')


