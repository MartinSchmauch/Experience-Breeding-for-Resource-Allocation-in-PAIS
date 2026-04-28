"""Factory for generating resources from experience data."""

from typing import List
from .resource import Resource
from ..experience.store import ExperienceStore


class ResourceFactory:
    """
    Factory for creating Resource instances from experience store data.
    
    Handles capability mapping from experience counts to capability levels.
    """        
    
    def create_resources(self, experience_store: ExperienceStore) -> List[Resource]:
        """
        Create Resource instances from experience store data.
        
        Args:
            experience_store: Loaded ExperienceStore
            
        Returns:
            List of Resource instances with store reference for capability queries
        """
        # Get all unique resources
        resource_ids = experience_store.get_all_resource_ids()
        
        print(f"Creating {len(resource_ids)} resources from experience store")
        
        resources = []
        for res_id in resource_ids:
            resource = Resource(
                id=res_id,
                name=res_id,  # Use ID as name (no HR data)
                experience_profile_id=res_id,
                experience_store=experience_store
            )
            resources.append(resource)
        
        return resources