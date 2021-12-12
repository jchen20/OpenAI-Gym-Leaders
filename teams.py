# Format: [# pokemon]_team_[1-indexed pokemon from poke env example]


one_team_1 = """
Goodra (M) @ Assault Vest
Ability: Sap Sipper
EVs: 248 HP / 252 SpA / 8 Spe
Modest Nature
IVs: 0 Atk
- Dragon Pulse
- Flamethrower
- Sludge Wave
- Thunderbolt
"""

two_team_1_2 = """
Goodra (M) @ Assault Vest
Ability: Sap Sipper
EVs: 248 HP / 252 SpA / 8 Spe
Modest Nature
IVs: 0 Atk
- Dragon Pulse
- Flamethrower
- Sludge Wave
- Thunderbolt

Sylveon (M) @ Leftovers
Ability: Pixilate
EVs: 248 HP / 244 Def / 16 SpD
Calm Nature
IVs: 0 Atk
- Hyper Voice
- Mystical Fire
- Protect
- Wish
"""

three_team_1_2_4 = """
Goodra (M) @ Assault Vest
Ability: Sap Sipper
EVs: 248 HP / 252 SpA / 8 Spe
Modest Nature
IVs: 0 Atk
- Dragon Pulse
- Flamethrower
- Sludge Wave
- Thunderbolt

Sylveon (M) @ Leftovers
Ability: Pixilate
EVs: 248 HP / 244 Def / 16 SpD
Calm Nature
IVs: 0 Atk
- Hyper Voice
- Mystical Fire
- Protect
- Wish

Toxtricity (M) @ Throat Spray
Ability: Punk Rock
EVs: 4 Atk / 252 SpA / 252 Spe
Rash Nature
- Overdrive
- Boomburst
- Shift Gear
- Fire Punch
"""

three_team_1_4_5 = """
Goodra (M) @ Assault Vest
Ability: Sap Sipper
EVs: 248 HP / 252 SpA / 8 Spe
Modest Nature
IVs: 0 Atk
- Dragon Pulse
- Flamethrower
- Sludge Wave
- Thunderbolt

Toxtricity (M) @ Throat Spray
Ability: Punk Rock
EVs: 4 Atk / 252 SpA / 252 Spe
Rash Nature
- Overdrive
- Boomburst
- Shift Gear
- Fire Punch

Seismitoad (M) @ Leftovers
Ability: Water Absorb
EVs: 252 HP / 252 Def / 4 SpD
Relaxed Nature
- Stealth Rock
- Scald
- Earthquake
- Toxic
"""

six_team = """
Goodra (M) @ Assault Vest
Ability: Sap Sipper
EVs: 248 HP / 252 SpA / 8 Spe
Modest Nature
IVs: 0 Atk
- Dragon Pulse
- Flamethrower
- Sludge Wave
- Thunderbolt

Sylveon (M) @ Leftovers
Ability: Pixilate
EVs: 248 HP / 244 Def / 16 SpD
Calm Nature
IVs: 0 Atk
- Hyper Voice
- Mystical Fire
- Protect
- Wish

Sandaconda @ Focus Sash
Ability: Sand Spit
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Stealth Rock
- Glare
- Earthquake
- Rock Tomb

Toxtricity (M) @ Throat Spray
Ability: Punk Rock
EVs: 4 Atk / 252 SpA / 252 Spe
Rash Nature
- Overdrive
- Boomburst
- Shift Gear
- Fire Punch

Seismitoad (M) @ Leftovers
Ability: Water Absorb
EVs: 252 HP / 252 Def / 4 SpD
Relaxed Nature
- Stealth Rock
- Scald
- Earthquake
- Toxic

Corviknight (M) @ Leftovers
Ability: Pressure
EVs: 248 HP / 80 SpD / 180 Spe
Impish Nature
- Defog
- Brave Bird
- Roost
- U-turn
"""

wall_six_team = """
Blissey (M) @ Heavy-Duty Boots
Ability: Natural Cure
EVs: 248 HP / 252 Def / 8 SpD
Bold Nature
- Stealth Rock
- Soft-Boiled
- Seismic Toss
- Toxic

Corviknight (M) @ Leftovers
Ability: Pressure
EVs: 252 HP / 168 Def / 88 SpD
Relaxed Nature
- Defog
- Roost
- U-turn
- Brave Bird

Quagsire (M) @ Leftovers
Ability: Unaware
EVs: 252 HP / 252 Def / 4 SpD
Relaxed Nature
IVs: 0 Atk
- Scald
- Earthquake
- Recover
- Toxic

Clefable (M) @ Leftovers
Ability: Unaware
EVs: 252 HP / 252 Def / 4 SpD
Bold Nature
IVs: 0 Atk
- Aromatherapy
- Moonblast
- Wish
- Protect

Toxapex (M) @ Black Sludge
Ability: Regenerator
EVs: 252 HP / 252 Def / 4 SpD
Bold Nature
- Scald
- Recover
- Haze
- Knock Off

Hydreigon (M) @ Leftovers
Ability: Levitate
EVs: 164 HP / 92 SpA / 252 Spe
Timid Nature
- Taunt
- Dark Pulse
- Earth Power
- Roost
"""

random_pokemon_list = ["""
Weavile @ Heavy-Duty Boots
Ability: Pressure
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Swords Dance
- Triple Axel
- Knock Off
- Ice Shard
""", """
Blissey (M) @ Heavy-Duty Boots
Ability: Natural Cure
EVs: 248 HP / 252 Def / 8 SpD
Bold Nature
- Stealth Rock
- Soft-Boiled
- Seismic Toss
- Toxic
""",
"""
Corviknight (M) @ Leftovers
Ability: Pressure
EVs: 252 HP / 168 Def / 88 SpD
Relaxed Nature
- Defog
- Roost
- U-turn
- Brave Bird
""",
"""
Quagsire (M) @ Leftovers
Ability: Unaware
EVs: 252 HP / 252 Def / 4 SpD
Relaxed Nature
IVs: 0 Atk
- Scald
- Earthquake
- Recover
- Toxic
""",
"""
Clefable (M) @ Leftovers
Ability: Unaware
EVs: 252 HP / 252 Def / 4 SpD
Bold Nature
IVs: 0 Atk
- Aromatherapy
- Moonblast
- Wish
- Protect
""",
"""
Toxapex (M) @ Black Sludge
Ability: Regenerator
EVs: 252 HP / 252 Def / 4 SpD
Bold Nature
- Scald
- Recover
- Haze
- Knock Off
""",
"""
Hydreigon (M) @ Leftovers
Ability: Levitate
EVs: 164 HP / 92 SpA / 252 Spe
Timid Nature
- Taunt
- Dark Pulse
- Earth Power
- Roost
""",
"""
Goodra (M) @ Assault Vest
Ability: Sap Sipper
EVs: 248 HP / 252 SpA / 8 Spe
Modest Nature
IVs: 0 Atk
- Dragon Pulse
- Flamethrower
- Sludge Wave
- Thunderbolt
""",
"""
Sylveon (M) @ Leftovers
Ability: Pixilate
EVs: 248 HP / 244 Def / 16 SpD
Calm Nature
IVs: 0 Atk
- Hyper Voice
- Mystical Fire
- Protect
- Wish
""",
"""
Sandaconda @ Focus Sash
Ability: Sand Spit
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Stealth Rock
- Glare
- Earthquake
- Rock Tomb
""",
"""
Toxtricity (M) @ Throat Spray
Ability: Punk Rock
EVs: 4 Atk / 252 SpA / 252 Spe
Rash Nature
- Overdrive
- Boomburst
- Shift Gear
- Fire Punch
""",
"""
Seismitoad (M) @ Leftovers
Ability: Water Absorb
EVs: 252 HP / 252 Def / 4 SpD
Relaxed Nature
- Stealth Rock
- Scald
- Earthquake
- Toxic
""",
"""
Tyranitar @ Leftovers
Ability: Sand Stream
EVs: 252 HP / 4 Def / 252 SpD
Careful Nature
- Stealth Rock
- Rock Blast
- Earthquake
- Thunder Wave
""",
"""
Urshifu-Rapid-Strike @ Choice Band
Ability: Unseen Fist
EVs: 252 Atk / 4 Def / 252 Spe
Jolly Nature
- Surging Strikes
- Close Combat
- Aqua Jet
- U-turn
""",
"""
Rillaboom @ Life Orb
Ability: Grassy Surge
EVs: 252 Atk / 4 Def / 252 Spe
Adamant Nature
- Swords Dance
- Grassy Glide
- Knock Off
- Superpower
""",
"""
Heatran @ Leftovers
Ability: Flash Fire
EVs: 252 HP / 232 SpD / 24 Spe
Calm Nature
- Magma Storm
- Earth Power
- Taunt
- Toxic
""",
"""
Scizor @ Heavy-Duty Boots
Ability: Technician
EVs: 248 HP / 172 Def / 88 SpD
Impish Nature
- Swords Dance
- Bullet Punch
- Knock Off
- Roost
""",
"""
Kartana @ Choice Band
Ability: Beast Boost
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Leaf Blade
- Smart Strike
- Knock Off
- Sacred Sword
""",
"""
Hippowdon @ Leftovers
Ability: Sand Stream
EVs: 252 HP / 8 Atk / 248 SpD
Careful Nature
- Earthquake
- Slack Off
- Stealth Rock
- Toxic
""",
"""
Azumarill @ Sitrus Berry
Ability: Huge Power
EVs: 4 HP / 252 Atk / 252 Spe
Adamant Nature
- Belly Drum
- Play Rough
- Aqua Jet
- Knock Off
""",
"""
Ferrothorn @ Leftovers
Ability: Iron Barbs
EVs: 252 HP / 252 Def / 4 SpD
Impish Nature
- Spikes
- Knock Off
- Leech Seed
- Power Whip
""",
"""
Magnezone @ Leftovers
Ability: Magnet Pull
EVs: 4 HP / 252 Def / 252 Spe
Timid Nature
- Iron Defense
- Body Press
- Thunderbolt
- Toxic
"""
]