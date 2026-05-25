from enum import Enum, auto, IntFlag


class Zone(Enum):
    LIBRARY = auto()
    HAND = auto()
    BATTLEFIELD = auto()
    GRAVEYARD = auto()
    EXILE = auto()
    STACK = auto()
    COMMAND = auto()


class Color(IntFlag):
    NONE = 0
    W = 1
    U = 2
    B = 4
    R = 8
    G = 16


class CardType(IntFlag):
    NONE = 0
    LAND = 1
    CREATURE = 2
    ARTIFACT = 4
    ENCHANTMENT = 8
    INSTANT = 16
    SORCERY = 32
    PLANESWALKER = 64
    TRIBAL = 128


class Subtype(Enum):
    # creature types we care about
    HUMAN = auto()
    SOLDIER = auto()
    KNIGHT = auto()
    ELEPHANT = auto()
    GOBLIN = auto()
    WIZARD = auto()
    DRAGON = auto()
    BIRD = auto()
    VAMPIRE = auto()
    PHOENIX = auto()
    ELEMENTAL = auto()
    VIASHINO = auto()
    SHAMAN = auto()
    CAT = auto()
    WARRIOR = auto()
    PIRATE = auto()
    # land
    PLAINS = auto()
    MOUNTAIN = auto()
    # other
    SAGA = auto()
    AURA = auto()
    EQUIPMENT = auto()
    AJANI = auto()


class Keyword(Enum):
    FLYING = auto()
    FIRST_STRIKE = auto()
    DOUBLE_STRIKE = auto()
    LIFELINK = auto()
    VIGILANCE = auto()
    MENACE = auto()
    HASTE = auto()
    INDESTRUCTIBLE = auto()
    DEFENDER = auto()
    TRAMPLE = auto()
    REACH = auto()
    DEATHTOUCH = auto()
    HEXPROOF = auto()


class Phase(Enum):
    BEGIN = auto()
    MAIN1 = auto()
    COMBAT = auto()
    MAIN2 = auto()
    END = auto()


class Step(Enum):
    UNTAP = auto()
    UPKEEP = auto()
    DRAW = auto()
    PRECOMBAT_MAIN = auto()
    BEGIN_COMBAT = auto()
    DECLARE_ATTACKERS = auto()
    DECLARE_BLOCKERS = auto()
    FIRST_STRIKE_DAMAGE = auto()
    COMBAT_DAMAGE = auto()
    END_COMBAT = auto()
    POSTCOMBAT_MAIN = auto()
    END_STEP = auto()
    CLEANUP = auto()


class TriggerEvent(Enum):
    ETB = auto()                # enters the battlefield
    LTB = auto()                # leaves battlefield
    CAST = auto()               # spell cast
    DEALS_DAMAGE_OPP = auto()   # damage dealt to opponent
    ATTACKS = auto()            # creature attacks
    BLOCKS = auto()
    BECOMES_TAPPED = auto()
    DIES = auto()
    DRAWS_CARD = auto()
    LIFE_LOST_OPP = auto()      # opponent lost life this turn (for spectacle gating)
    UPKEEP = auto()
    END_STEP = auto()
    SAGA_CHAPTER = auto()
    BEGIN_COMBAT = auto()
