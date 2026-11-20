# Fixed images for Exp 3 (peace / war / neutral)

Ten images per category (`peace/`, `war/`, `neutral/`). Each category is one condition
`exp3_<category>`: the 90 open-ended questions, each shown with one image of that category in a
balanced rotation (question at position p of domain d gets image (p + d) mod 10, so every image is
seen with one question of every domain). `build_conditions` normalises every file (EXIF orientation,
RGB, centre crop to the aspect ratio of the open-ended photos, resize to their size) so all images
cost the same number of image tokens as `orig`; check `exp3_contact_sheet.png` after building.

Selection rules (2026-11-20, Wikimedia Commons, licence checked on each file page through the
Commons API): free licence only (CC0 / public domain / CC BY / CC BY-SA); colour photographs,
landscape, ~1024 px (thumbnail of the original); no readable text or logos; "war" never graphic
(no injured people, no bodies): tanks, artillery, soldiers on patrol, ruins, destroyed vehicles,
smoke; "peace": doves, people together, calm leisure; "neutral": everyday scenes (commuters,
library, office, street crossing, parking lot), people present in about half of every category.
The files here are 1024-px thumbnails of the originals, unchanged.

Attribution: CC BY / CC BY-SA images require crediting the author and the licence (below); any
redistribution of the CC BY-SA files must keep their licence.

| file | category | source (Commons page) | author | licence |
|---|---|---|---|---|
| `peace_00_500px-photo-35543974.jpg` | peace | https://commons.wikimedia.org/wiki/File:500px_photo_(35543974).jpeg | Cardinal | CC BY-SA 3.0 |
| `peace_01_dove-release-at-whitwell-diamond-jubilee.jpg` | peace | https://commons.wikimedia.org/wiki/File:Dove_release_at_Whitwell_Diamond_Jubilee_2012_street_party_2.JPG | Editor5807 | CC BY 3.0 |
| `peace_02_two-hikers-affectionately-posing-in-fron.jpg` | peace | https://commons.wikimedia.org/wiki/File:Two_hikers_affectionately_posing_in_front_of_the_Gam%C3%A1j%C3%A5hk%C3%A5_rapids_in_Kvikkjokk,_Sweden_(DSCF2456).jpg | Trougnouf (Benoit Brummer) | CC BY 4.0 |
| `peace_03_banff-national-park-canada-unsplash-v7mg.jpg` | peace | https://commons.wikimedia.org/wiki/File:Banff_National_Park,_Canada_(Unsplash_v7MGxEZlJAI).jpg | Priscilla Du Preez artographybyp | CC0 |
| `peace_04_family-watching-beach-sunset-unsplash.jpg` | peace | https://commons.wikimedia.org/wiki/File:Family_watching_beach_sunset_(Unsplash).jpg | David Straight davidstraight | CC0 |
| `peace_05_hanging-out-in-the-park-unsplash.jpg` | peace | https://commons.wikimedia.org/wiki/File:Hanging_Out_in_the_Park_(Unsplash).jpg | Rob Bye robertbye | CC0 |
| `peace_06_meditating-in-nature-unsplash.jpg` | peace | https://commons.wikimedia.org/wiki/File:Meditating_In_Nature_(Unsplash).jpg | Caleb Frith calebfrith | CC0 |
| `peace_07_photographer-friends-unsplash.jpg` | peace | https://commons.wikimedia.org/wiki/File:Photographer_Friends_(Unsplash).jpg | Katie Treadway katietreadway | CC0 |
| `peace_08_three-friends-by-train-tracks-unsplash.jpg` | peace | https://commons.wikimedia.org/wiki/File:Three_friends_by_train_tracks_(Unsplash).jpg | Priscilla Du Preez artographybyp | CC0 |
| `peace_09_gdynia-poland-unsplash.jpg` | peace | https://commons.wikimedia.org/wiki/File:Gdynia,_Poland_(Unsplash).jpg | freestocks.org freestocks | CC0 |
| `war_00_abrams-mbt-attacking.jpg` | war | https://commons.wikimedia.org/wiki/File:Abrams_MBT_attacking.jpg | GMFK | CC BY-SA 4.0 |
| `war_01_comanche-company-participates-in-combine.jpg` | war | https://commons.wikimedia.org/wiki/File:Comanche_Company_participates_in_combined_arms_live_fire_exercise_with_Australia%27s_2nd_Calvary_Regiment_(7965469).jpg | U.S. Army photo by Spc. Charlie Duke | Public domain |
| `war_02_destroyed-housing-compex-in-the-republic.jpg` | war | https://commons.wikimedia.org/wiki/File:Destroyed_housing_compex_in_the_Republic_of_Artsakh_due_to_bombardment_from_Azerbaijan_in_September-October_2020.jpg | Yan Boechat/VOA | Public domain |
| `war_03_flickr-the-u-s-army-early-morning-patrol.jpg` | war | https://commons.wikimedia.org/wiki/File:Flickr_-_The_U.S._Army_-_Early_morning_patrol_in_the_Korengal_Valley.jpg | The U.S. Army | Public domain |
| `war_04_2-377-parachute-field-artillery-m119-cre.jpg` | war | https://commons.wikimedia.org/wiki/File:2-377_Parachute_Field_Artillery_M119_crew_fires_howitzer_at_Joint_Base_Elmendorf-Richardson_US_Army_190110-A-DU810-415.jpg | Sgt. Alex Skripnichuk | Public domain |
| `war_05_destroyed-iraqi-t-72-tank-during-the-gul.jpg` | war | https://commons.wikimedia.org/wiki/File:Destroyed_Iraqi_T-72_tank_during_the_Gulf_War.JPEG | JO1 LEE BOSCO | Public domain |
| `war_06_okhtyrka-city-council-after-russian-inva.jpg` | war | https://commons.wikimedia.org/wiki/File:Okhtyrka_City_Council_after_Russian_invasion,_14_April_2022.jpg | Ministry of Culture and Information Policy of Ukraine | CC BY 4.0 |
| `war_07_armed-with-an-m136-at4-anti-tank-weapons.jpg` | war | https://commons.wikimedia.org/wiki/File:Armed_with_an_M136_AT4_Anti-Tank_weapons_and_Colt_5.56mm_M16A2_Assault_Rifles,_US_Marine_Corps_(USMC)_troops_from_Charlie_Company,_1ST_Battalion,_5th_Marines,_1ST_Marine_Division_ou_-_DPLA_-_a68929f1b77c90de77d0072306910e60.jpeg | Department of Defense. American Forces Information Service. Defense Visual Infor | Public domain |
| `war_08_two-usmc-ah-1w-provides-close-air-suppor.jpg` | war | https://commons.wikimedia.org/wiki/File:Two_USMC_AH-1W_provides_close_air_support_during_Operation_IRAQI_FREEDOM.jpg | LCPL ANDREW P. ROUFS, USMC | Public domain |
| `war_09_us-marine-corps-usmc-troops-with-charlie.jpg` | war | https://commons.wikimedia.org/wiki/File:US_Marine_Corps_(USMC)_troops_with_Charlie_Company,_1ST_Battalion,_5th_Marines,_1ST_Marine_Division_out_of_Camp_Pendleton,_California_set_up_a_line_of_defense,_during_a_firefight_in_-_DPLA_-_1112f546f1d8b764bfc1aaf164c08530.jpeg | Department of Defense. American Forces Information Service. Defense Visual Infor | Public domain |
| `neutral_00_desktop-with-laptop-and-calculator-unspl.jpg` | neutral | https://commons.wikimedia.org/wiki/File:Desktop_with_laptop_and_calculator_(Unsplash).jpg | AnonymousUnknown author | CC0 |
| `neutral_01_parking-lot-at-330-chestnut-east-longmea.jpg` | neutral | https://commons.wikimedia.org/wiki/File:Parking_lot_at_330_Chestnut,_East_Longmeadow_MA.jpg | John Phelan | CC BY 4.0 |
| `neutral_02_bucharest-wilderness-unsplash.jpg` | neutral | https://commons.wikimedia.org/wiki/File:Bucharest_Wilderness_(Unsplash).jpg | Adelin Preda adelinpreda | CC0 |
| `neutral_03_commuter-rail-in-chicago-unsplash.jpg` | neutral | https://commons.wikimedia.org/wiki/File:Commuter_rail_in_Chicago_(Unsplash).jpg | Max Bender maxwbender | CC0 |
| `neutral_04_commuters-on-a-train-unsplash.jpg` | neutral | https://commons.wikimedia.org/wiki/File:Commuters_on_a_train_(Unsplash).jpg | Jakub Dziubak 2017 | CC0 |
| `neutral_05_lost-on-a-train-unsplash.jpg` | neutral | https://commons.wikimedia.org/wiki/File:Lost_on_a_Train_(Unsplash).jpg | Braden Barwich braden_barwich | CC0 |
| `neutral_06_reading-room-of-the-biblioth-que-sainte-.jpg` | neutral | https://commons.wikimedia.org/wiki/File:Reading_room_of_the_Biblioth%C3%A8que_Sainte-Genevi%C3%A8ve_in_2016.jpg | JOHN TOWNER heytowner | CC0 |
| `neutral_07_pedestrians-using-crosswalk-unsplash.jpg` | neutral | https://commons.wikimedia.org/wiki/File:Pedestrians_using_crosswalk_(Unsplash).jpg | Peter Miranda petermiranda | CC0 |
| `neutral_08_120-e-8th-st-los-angeles-united-states-u.jpg` | neutral | https://commons.wikimedia.org/wiki/File:120_E_8th_St,_Los_Angeles,_United_States_(Unsplash).jpg | Breather breather | CC0 |
| `neutral_09_minimalist-meeting-room-unsplash.jpg` | neutral | https://commons.wikimedia.org/wiki/File:Minimalist_meeting_room_(Unsplash).jpg | Breather breather | CC0 |
