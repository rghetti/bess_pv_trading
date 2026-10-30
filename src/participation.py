"""Auction subsets preserve market chronology; participation does not reorder gates."""
from itertools import combinations

AUCTION_ORDER = ("DA", "IDA1", "IDA2", "IDA3")


def auction_subset(value, label, *, allow_empty=False):
    """Validate explicit unique names and normalize them to market order."""
    # Reject input that violates this condition.
    if (not isinstance(value, list) or (not value and not allow_empty)
            or any(not isinstance(item, str) or item not in AUCTION_ORDER for item in value)
            or len(value) != len(set(value))):
        raise ValueError(f"{label} requires a unique {'possibly empty ' if allow_empty else 'nonempty '}list of DA/IDA1/IDA2/IDA3")
    return tuple(auction for auction in AUCTION_ORDER if auction in value)


def participation_profiles(settings):
    """Return every allowed nonempty subset, or explicitly selected profiles.

    Four optional auctions give 15 nonempty profiles. Requiring DA gives eight
    profiles by independently including/excluding each of the three IDA auctions.
    """
    candidates = auction_subset(settings.get("candidate_auctions", list(AUCTION_ORDER)),
                                "participation_study.candidate_auctions")
    required = auction_subset(settings.get("required_auctions", []),
                              "participation_study.required_auctions", allow_empty=True)
    # Enforce the input contract: Required auctions must belong to candidate_auctions.
    if not set(required).issubset(candidates):
        raise ValueError("Required auctions must belong to candidate_auctions")
    explicit = settings.get("profiles")
    # When `explicit is None`, set `optional` using `[auction for auction in candidates if auction not in required]`.
    if explicit is None:
        optional = [auction for auction in candidates if auction not in required]
        profiles = [tuple(auction for auction in AUCTION_ORDER if auction in (*required, *subset))
            for size in range(len(optional) + 1) for subset in combinations(optional, size)
            if required or subset]
    else:
        # Enforce the input contract: participation_study.profiles must be null or a nonempty list of auction lists.
        if not isinstance(explicit, list) or not explicit:
            raise ValueError("participation_study.profiles must be null or a nonempty list of auction lists")
        profiles = [auction_subset(profile, "participation_study.profiles entry") for profile in explicit]
    # Enforce the input contract: Participation profiles must be unique.
    if len(profiles) != len(set(profiles)):
        raise ValueError("Participation profiles must be unique")
    # Enforce the input contract: Every profile must include required_auctions and use only candidate_auctions.
    if any(not set(required).issubset(profile) or not set(profile).issubset(candidates) for profile in profiles):
        raise ValueError("Every profile must include required_auctions and use only candidate_auctions")
    return tuple(profiles)


def profile_id(profile):
    """Use a stable identifier usable as a folder and a CSV key."""
    return "+".join(profile)
