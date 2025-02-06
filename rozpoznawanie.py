import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

model_path = "mushroom_classifier_resnet50.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 215)
state_dict = torch.load(model_path, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

classes = [
    "almond_mushroom", "amanita_gemmata", "amethyst_chanterelle", "amethyst_deceiver", "aniseed_funnel",
    "ascot_hat", "bay_bolete", "bearded_milkcap", "beechwood_sickener", "beefsteak_fungus",
    "birch_polypore", "birch_woodwart", "bitter_beech_bolete", "bitter_bolete", "black_bulgar",
    "black_morel", "blackening_brittlegill", "blackening_polypore", "blackening_waxcap", "blue_roundhead",
    "blushing_bracket", "blushing_rosette", "blushing_wood_mushroom", "bovine_bolete", "bronze_bolete",
    "brown_birch_bolete", "brown_rollrim", "bruising_webcap", "butter_cap", "cauliflower_fungus",
    "cedarwood_waxcap", "chanterelle", "charcoal_burner", "chestnut_bolete", "chicken_of_the_woods",
    "cinnamon_bracket", "clouded_agaric", "clustered_domecap", "common_bonnet", "common_inkcap",
    "common_morel", "common_puffball", "common_rustgill", "crimped_gill", "crimson_waxcap",
    "cucumber_cap", "curry_milkcap", "deadly_fibrecap", "deadly_webcap", "deathcap",
    "deer_shield", "destroying_angel", "devils_bolete", "dog_stinkhorn", "dryads_saddle",
    "dusky_puffball", "dyers_mazegill", "earthballs", "egghead_mottlegill", "elfin_saddle",
    "fairy_ring_champignons", "false_chanterelle", "false_deathcap", "false_morel", "false_saffron_milkcap",
    "fenugreek_milkcap", "field_blewit", "field_mushroom", "fleecy_milkcap", "fly_agaric",
    "fools_funnel", "fragrant_funnel", "freckled_dapperling", "frosted_chanterelle", "funeral_bell",
    "geranium_brittlegill", "giant_funnel", "giant_puffball", "glistening_inkcap", "golden_bootleg",
    "golden_scalycap", "golden_waxcap", "greencracked_brittlegill", "grey_knight", "grey_spotted_amanita",
    "grisettes", "hairy_curtain_crust", "heath_waxcap", "hedgehog_fungus", "hen_of_the_woods",
    "honey_fungus", "hoof_fungus", "horn_of_plenty", "horse_mushroom", "inky_mushroom",
    "jelly_ears", "jelly_tooth", "jubilee_waxcap", "king_alfreds_cakes", "larch_bolete",
    "leccinum_albostipitatum", "liberty_cap", "lilac_bonnet", "lilac_fibrecap", "lions_mane",
    "lurid_bolete", "macro_mushroom", "magpie_inkcap", "meadow_waxcap", "medusa_mushroom",
    "morel", "mosaic_puffball", "oak_bolete", "oak_mazegill", "oak_polypore",
    "ochre_brittlegill", "old_man_of_the_woods", "orange_birch_bolete", "orange_bolete", "orange_grisette",
    "orange_peel_fungus", "oyster_mushroom", "pale_oyster", "panthercap", "parasol",
    "parrot_waxcap", "pavement_mushroom", "penny_bun", "peppery_bolete", "pestle_puffball",
    "pine_bolete", "pink_waxcap", "plums_and_custard", "poison_pie", "poplar_bell",
    "poplar_fieldcap", "porcelain_fungus", "powdery_brittlegill", "purple_brittlegill", "red_belted_bracket",
    "red_cracking_bolete", "root_rot", "rooting_bolete", "rooting_shank", "rosy_bonnet",
    "ruby_bolete", "saffron_milkcap", "scaly_wood_mushroom", "scarlet_caterpillarclub", "scarlet_elfcup",
    "scarlet_waxcap", "scarletina_bolete", "semifree_morel", "sepia_bolete", "shaggy_bracket",
    "shaggy_inkcap", "shaggy_parasol", "shaggy_scalycap", "sheathed_woodtuft", "silky_rosegill",
    "silverleaf_fungus", "slender_parasol", "slimy_waxcap", "slippery_jack", "smoky_bracket",
    "snakeskin_grisette", "snowy_waxcap", "spectacular_rustgill", "splendid_waxcap", "splitgill",
    "spotted_toughshank", "spring_fieldcap", "st_georges_mushroom", "stinkhorn", "stinking_dapperling",
    "stubble_rosegill", "stump_puffball", "suede_bolete", "sulphur_tuft", "summer_bolete",
    "tawny_funnel", "tawny_grisette", "terracotta_hedgehog", "the_blusher", "the_deceiver",
    "the_goblet", "the_miller", "the_prince", "the_sickener", "thimble_morel",
    "tripe_fungus", "trooping_funnel", "truffles", "tuberous_polypore", "turkey_tail",
    "velvet_shank", "vermillion_waxcap", "warted_amanita", "weeping_widow", "white_dapperling",
    "white_domecap", "white_false_death_cap", "white_fibrecap", "white_saddle", "winter_chanterelle",
    "wood_blewit", "wood_mushroom", "woodland_inkcap", "woolly_milkcap", "wrinkled_peach",
    "yellow_false_truffle", "yellow_foot_waxcap", "yellow_stagshorn", "yellow_stainer", "yellow_swamp_brittlegill"
]


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_top_3(image_path, model, classes):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)

    top_probs, top_idxs = torch.topk(probabilities, 3)
    top_probs = top_probs.cpu().numpy().flatten()
    top_idxs = top_idxs.cpu().numpy().flatten()

    predictions = [(classes[idx], prob) for idx, prob in zip(top_idxs, top_probs)]
    return predictions

image_path = "grzyby/kania.jpg"

predictions = predict_top_3(image_path, model, classes)

for class_name, probability in predictions:
    print(f"Class: {class_name}, Probability: {probability:.4f}")